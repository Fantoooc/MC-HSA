use std::{array::from_fn, collections::HashMap, env, error::Error, fs::File, io::Cursor};
use base64::Engine;
use bytes::Bytes;
use gif::{Encoder, Frame};
use image::{GenericImageView, ImageReader};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::f32::consts::PI as PI_32;
use reqwest::{self, Client};

const TEX: usize = 8;
const FACE: usize = TEX * TEX * 4;

const X: [f32; 6] = [0.0, 0.0, 1.0, 0.0, -1.0, 0.0];
const Y: [f32; 6] = [1.0, -1.0, 0.0, 0.0, 0.0, 0.0];
const Z: [f32; 6] = [0.0, 0.0, 0.0, 1.0, 0.0, -1.0];

const CORNERS: [f32; 72] = [ -0.5, 0.5, -0.5, 0.5, 0.5, -0.5, 0.5, 0.5, 0.5, -0.5, 0.5, 0.5 /* Top */, -0.5, -0.5, -0.5, 0.5, -0.5, -0.5, 0.5, -0.5, 0.5, -0.5, -0.5, 0.5 /* Bottom */, 0.5, 0.5, 0.5, 0.5, 0.5, -0.5, 0.5, -0.5, -0.5, 0.5, -0.5, 0.5 /* Right */, -0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, -0.5, 0.5, -0.5, -0.5, 0.5 /* Front */, -0.5, 0.5, -0.5, -0.5, 0.5, 0.5, -0.5, -0.5, 0.5, -0.5, -0.5, -0.5 /* Left */, 0.5, 0.5, -0.5, -0.5, 0.5, -0.5, -0.5, -0.5, -0.5, 0.5, -0.5, -0.5 /* Back */ ];
const CORNERS_SCALED: [f32; 72] = [ -0.56, 0.56, -0.56, 0.56, 0.56, -0.56, 0.56, 0.56, 0.56, -0.56, 0.56, 0.56 /* Top */, -0.56, -0.56, -0.56, 0.56, -0.56, -0.56, 0.56, -0.56, 0.56, -0.56, -0.56, 0.56 /* Bottom */, 0.56, 0.56, 0.56, 0.56, 0.56, -0.56, 0.56, -0.56, -0.56, 0.56, -0.56, 0.56 /* Right */, -0.56, 0.56, 0.56, 0.56, 0.56, 0.56, 0.56, -0.56, 0.56, -0.56, -0.56, 0.56 /* Front */, -0.56, 0.56, -0.56, -0.56, 0.56, 0.56, -0.56, -0.56, 0.56, -0.56, -0.56, -0.56 /* Left */, 0.56, 0.56, -0.56, -0.56, 0.56, -0.56, -0.56, -0.56, -0.56, 0.56, -0.56, -0.56 /* Back */ ];

#[derive(Debug)]
struct Args
{
    username: String,
    output: String,
    frames: u16,
    canvas_size: u16,
    scale: f32,
    xoff: f32,
    yoff: f32,
    pitch: f32
}

async fn fetch(username: &String)
-> Result<Bytes, Box<dyn Error>>
{
    let client = Client::new();
    let req = client.get(&format!("https://api.mojang.com/users/profiles/minecraft/{username}"))
        .send()
        .await?
        .error_for_status()?;
    if req.status() == reqwest::StatusCode::NO_CONTENT { return Err("Player not found".into()); }

    let res: serde_json::Value = req.json().await?;
    let id = res["id"].as_str().expect("No id found");
    println!("Parsed id from {username}");

    let req = client.get(&format!("https://sessionserver.mojang.com/session/minecraft/profile/{id}"))
        .send()
        .await?
        .error_for_status()?;
    if req.status() == reqwest::StatusCode::NO_CONTENT { return Err("Sessionserver profile not found".into()); }
    println!("Parsed sessionserver profile");

    let res: serde_json::Value = req.json().await?;

    let textures: serde_json::Value = serde_json::from_slice(&base64::prelude::BASE64_STANDARD.decode(&res["properties"][0]["value"].as_str().expect("Wrong b64 string")).expect("Failed to parse b64 string"))?;

    let req = client.get(textures["textures"]["SKIN"]["url"].as_str().ok_or("fuck")?).send().await?.error_for_status()?;
    if req.status() == reqwest::StatusCode::NO_CONTENT { return Err("Failed to fetch skin".into()); }

    let res = req.bytes().await?;

    println!("Parsed texture bytes");
    Ok(res)
}

fn decode(data: Bytes)
-> Result<(Vec<u8>, usize, usize), String>
{
    if !data.starts_with(b"\x89PNG\r\n\x1a\n") {
        return Err("Not a png file has been downloaded".to_owned());
    }

    let image = ImageReader::with_format(Cursor::new(data), image::ImageFormat::Png).decode().map_err(|e| e.to_string())?;
    let (width, height) = image.dimensions();
    Ok((image.to_rgba8().into_raw(), width as usize, height as usize))
}

fn extract(pixels: &[u8], width: usize, height: usize, x0: usize, y0: usize)
-> Result<[u8; FACE], String>
{
    if x0 + TEX > width || y0 + TEX > height {
        return Err("Failed to extract pixels".to_owned());
    }
    let mut out = [0u8; FACE];
    for row in 0..TEX {
        let s = ((y0 + row) * width + x0) * 4;
        out[row * TEX * 4..(row + 1) * TEX * 4].copy_from_slice(&pixels[s..s + TEX * 4]);
    }
    Ok(out)
}

fn build_faces(pixels: &[u8], width: usize, height: usize)
-> Result<([u8; FACE * 6], Option<[u8; FACE * 6]>), String>
{
    const BASE: [usize; 12] = [8, 0, 16, 0, 16, 8, 8, 8, 0, 8, 24, 8];
    const HAT:  [usize; 12] = [40, 0, 48, 0, 48, 8, 40, 8, 32, 8, 56, 8];

    let mut tex = [0u8; FACE * 6];
    let mut hat = [0u8; FACE * 6];

    for i in 0..6 {
        tex[i * FACE..(i + 1) * FACE].copy_from_slice(&extract(pixels, width, height, BASE[2 * i], BASE[2 * i + 1])?);
        hat[i * FACE..(i + 1) * FACE].copy_from_slice(&extract(pixels, width, height, HAT[2 * i], HAT[2 * i + 1])?);
    }
    let has_hat = hat.chunks_exact(4).any(|p| p[3] > 10);
    Ok((tex, has_hat.then_some(hat)))
}

fn rotate_points(yaw: f32, pitch: f32)
-> [f32; 6]
{
    let (sy, cy) = yaw.sin_cos();
    let (sp, cp) = pitch.sin_cos();

    // let x1: [f32; 6] = from_fn(|i| X[i] * cy + Z[i] * sy);
    let z1: [f32; 6] = from_fn(|i| Z[i] * cy - X[i] * sy);
    // let y1: [f32; 6] = from_fn(|i| Y[i] * cp - z1[i] * sp);
    let z2: [f32; 6] = from_fn(|i| Y[i] * sp + z1[i] * cp);

    // (x1, y1, z2)
    z2
}

fn rotate_corners(yaw: f32, pitch: f32, scaled: bool)
-> ([f32; 24], [f32; 24], [f32; 24])
{
    let (sy, cy) = yaw.sin_cos();
    let (sp, cp) = pitch.sin_cos();

    let arr = if !scaled { CORNERS } else { CORNERS_SCALED };

    let x1: [f32; 24] = from_fn(|i| arr[i * 3] * cy + arr[2 + i * 3] * sy);
    let z1: [f32; 24] = from_fn(|i| arr[2 + i * 3] * cy - arr[i * 3] * sy);
    let y1: [f32; 24] = from_fn(|i| arr[1 + i * 3] * cp - z1[i] * sp);
    let z2: [f32; 24] = from_fn(|i| arr[1 + i * 3] * sp + z1[i] * cp);

    return (x1, y1, z2)
}

fn project(x: [f32; 24], y: [f32; 24], cx: f32, cy: f32, scale: f32, xoff: f32, yoff: f32)
-> ([f32; 24], [f32; 24])
{
    let px: [f32; 24] = from_fn(|i| cx + (x[i] - xoff) * scale);
    let py: [f32; 24] = from_fn(|i| cy - (y[i] - yoff) * scale);

    (px, py)
}

fn draw_faces(pixels: &mut [u8], tex: &[u8], yaw: f32, pitch: f32, scaled: bool, args: &Args) {
    let size_i = args.canvas_size as i32;
    let half = (args.canvas_size as f32) / 2.0;

    let rn_z = rotate_points(yaw, pitch);
    let (rc_x, rc_y, rc_z) = rotate_corners(yaw, pitch, scaled);
    let (pc_x, pc_y) = project(rc_x, rc_y, half, half, args.scale, args.xoff, args.yoff);

    let mut items = [(0.0f32, 0usize); 6];
    let mut n = 0;

    for (i, (&rn, chunk)) in rn_z.iter().zip(rc_z.chunks_exact(4)).enumerate() {
        if rn > 0.001 {
            let sum_z = chunk[0] + chunk[1] + chunk[2] + chunk[3];

            let mut j = n;
            while j > 0 && items[j - 1].0 > sum_z {
                items[j] = items[j - 1];
                j -= 1;
            }
            items[j] = (sum_z, i);
            n += 1;
        }
    }

    for &(_, i) in &items[..n] {
        let b = i * 4;
        let face_tex = &tex[i * FACE..(i + 1) * FACE];

        let (x0, y0) = (pc_x[b], pc_y[b]);
        let (ux, uy) = (pc_x[b + 1] - x0, pc_y[b + 1] - y0);
        let (vx, vy) = (pc_x[b + 3] - x0, pc_y[b + 3] - y0);

        let det = ux * vy - uy * vx;
        if det.abs() < 1e-9 { continue }
        let inv_det = 1.0 / det;

        let (xs, ys) = (&pc_x[b..b + 4], &pc_y[b..b + 4]);
        let minx = (xs.iter().copied().fold(f32::INFINITY, f32::min).floor() as i32 - 1).max(0);
        let maxx = (xs.iter().copied().fold(f32::NEG_INFINITY, f32::max).ceil() as i32 + 1).min(size_i - 1);
        let miny = (ys.iter().copied().fold(f32::INFINITY, f32::min).floor() as i32 - 1).max(0);
        let maxy = (ys.iter().copied().fold(f32::NEG_INFINITY, f32::max).ceil() as i32 + 1).min(size_i - 1);

        for py in miny..=maxy {
            for px in minx..=maxx {
                let dx = px as f32 + 0.5 - x0;
                let dy = py as f32 + 0.5 - y0;
                let u = (dx * vy - dy * vx) * inv_det;
                let v = (ux * dy - uy * dx) * inv_det;
                if !(0.0..1.0).contains(&u) || !(0.0..1.0).contains(&v) { continue; }

                let tx = ((u * TEX as f32) as usize).min(TEX - 1);
                let ty = ((v * TEX as f32) as usize).min(TEX - 1);
                let t = (ty * TEX + tx) * 4;
                let a = face_tex[t + 3] as f32 / 255.0;

                let p = ((py * size_i + px) * 4) as usize;
                for c in 0..3 {
                    pixels[p + c] = (face_tex[t + c] as f32 * a + pixels[p + c] as f32 * (1.0 - a) + 0.5) as u8;
                }
            }
        }
    }
}

fn render_frame(id: u16, args: &Args, tex: &[u8], hat: Option<&[u8]>)
-> Vec<u8>
{
    let yaw = 2.0 * (id as f32) * PI_32 / (args.frames as f32);
    let size = args.canvas_size as usize;

    let mut pixels = [54u8, 57, 63, 255].repeat(size * size);
    draw_faces(&mut pixels, tex, yaw, args.pitch, false, args);
    if let Some(hat) = hat { draw_faces(&mut pixels, hat, yaw, args.pitch, true, args);}

    pixels
}

fn write(args: &Args, frames: &[Vec<u8>])
-> Result<(), String>
{
    if frames.is_empty() { return Err("Frames is empty.".to_owned()) }
    let size = args.canvas_size;

    let all = frames.concat();
    let nq = color_quant::NeuQuant::new(5, 256, &all);
    let pallete = nq.color_map_rgb();

    let mut cache: HashMap<[u8; 3], u8> = HashMap::new();
    let mut last_key = [0u8; 3];
    let mut last_idx = nq.index_of(&[0, 0, 0, 255]) as u8;

    let file = File::create(&args.output).map_err(|e| e.to_string())?;
    let mut encoder = Encoder::new(file, size, size, &pallete).map_err(|e| e.to_string())?;
    encoder.set_repeat(gif::Repeat::Infinite).map_err(|e| e.to_string())?;

    for f in frames {
        let indices: Vec<u8> = f.chunks_exact(4).map(|p| {
            let key = [p[0], p[1], p[2]];
            if key == last_key { return last_idx; }
            let idx = *cache.entry(key).or_insert_with(|| nq.index_of(&[p[0], p[1], p[2], 255]) as u8);
            last_key = key;
            last_idx = idx;
            idx
        }).collect();
        let mut frame = Frame::from_indexed_pixels(size, size, indices, None);
        frame.delay = 6;
        encoder.write_frame(&frame).map_err(|e| e.to_string())?;
    }

    Ok(())
}

#[tokio::main]
async fn main()
-> Result<(), String>
{
    let mut env_args = env::args().skip(1);

    let mut args = Args {
        username: env_args.next().ok_or("Usage: ./script nickname [output.gif] [--frames N] [--size PX] [--xoff f64] [--yoff f64] [--pitch i16]")?,
        output: "output.gif".to_owned(),
        frames: 36,
        canvas_size: 200,
        scale: 68.0,
        xoff: 0.0,
        yoff: 0.1,
        pitch: -0.31415927
    };

    while let (Some(i), Some(inext)) = (env_args.next(), env_args.next()) {
        match i.as_str() {
            "--output" => { args.output = format!("{inext}.gif"); }
            "--frames" => { if let Some(frames) = inext.parse().ok() { args.frames = frames } else { println!("Передано неверное значение frames(используется стандартное: 36)") } }
            "--size" => { if let Some(size) = inext.parse().ok() { args.canvas_size = size; args.scale = (size as f32) * 0.34 } else { println!("Передано неверное значение canvas_size(используется стандартное: 200)") } }
            "--xoff" => { if let Some(xoff) = inext.parse().ok() { args.xoff = xoff } else { println!("Передано неверное значение xoff(используется стандартное: 0.0)") } }
            "--yoff" => { if let Some(yoff) = inext.parse().ok() { args.yoff = yoff } else { println!("Передано неверное значение yoff(используется стандартное: 0.1)") } }
            "--pitch" => { if let Some(pitch) = inext.parse::<f32>().ok() { args.pitch = pitch.to_radians() } else { println!("Передано неверное значение pitch(используется стандартное: -18)") } }
            _ => continue
        };
    }

    println!("{:#?}", args);

    let texture = fetch(&args.username).await.expect("Failed to parse");
    let (pixels, width, height) = decode(texture).expect("Failed to decode texture");

    let (tex, hat) = build_faces(&pixels, width, height)?;
    let hat = hat.as_ref().map(|h| &h[..]);
    let frames: Vec<_> = (0..args.frames).into_par_iter().map(|id| render_frame(id, &args, &tex, hat)).collect();

    write(&args, &frames)?;

    Ok(())
}