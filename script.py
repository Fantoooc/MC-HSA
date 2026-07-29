import sys
import json, base64
import math
import requests
from PIL import Image
import io

def fetch(username: str) -> bytes:
    headers = { "User-Agent": "minecraft-heads-spin-script/1.0"}
    try:
        print(f"Parsing {username}'s minecraft skin")

        req1 = requests.get(f"https://api.mojang.com/users/profiles/minecraft/{username}", headers = headers)
        uuid = req1.json()["id"]
        print(f"Profile id is: {uuid}")

        req2 = requests.get(f"https://sessionserver.mojang.com/session/minecraft/profile/{uuid}", headers = headers)
        value = req2.json()["properties"][0]["value"]

        textures = json.loads(base64.b64decode(value).decode("utf-8"))
        skin_url = textures["textures"]["SKIN"]["url"]

        req3 = requests.get(skin_url, headers = headers)
        return req3.content
    except Exception as e:
        raise ValueError(e)
    return

def decode_png(data: bytes):
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("Not a png file has been downloaded")

    img = Image.open(io.BytesIO(data)).convert("RGBA")
    width, height = img.size
    pixels = img.tobytes()

    return width, height, pixels

def get_texel(pixels, img_w, x, y):
    off = (y * img_w + x) * 4
    return pixels[off], pixels[off + 1], pixels[off + 2], pixels[off + 3]

def extract_face(pixels, img_w, x0, y0, size=8):
    face = []
    for y in range(size):
        row = []
        for x in range(size):
            row.append(get_texel(pixels, img_w, x0 + x, y0 + y))
        face.append(row)
    return face

def build_head_faces(pixels, img_w):
    def base(x0, y0):
        return extract_face(pixels, img_w, x0, y0, 8)

    faces = [
        dict(name="top", normal=(0, 1, 0),
             corners=[(-.5, .5, -.5), (.5, .5, -.5), (.5, .5, .5), (-.5, .5, .5)],
             tex=base(8, 0)),
        dict(name="bottom", normal=(0, -1, 0),
             corners=[(-.5, -.5, -.5), (.5, -.5, -.5), (.5, -.5, .5), (-.5, -.5, .5)],
             tex=base(16, 0)),
        dict(name="right", normal=(1, 0, 0),
             corners=[(.5, .5, .5), (.5, .5, -.5), (.5, -.5, -.5), (.5, -.5, .5)],
             tex=base(16, 8)),
        dict(name="front", normal=(0, 0, 1),
             corners=[(-.5, .5, .5), (.5, .5, .5), (.5, -.5, .5), (-.5, -.5, .5)],
             tex=base(8, 8)),
        dict(name="left", normal=(-1, 0, 0),
             corners=[(-.5, .5, -.5), (-.5, .5, .5), (-.5, -.5, .5), (-.5, -.5, -.5)],
             tex=base(0, 8)),
        dict(name="back", normal=(0, 0, -1),
             corners=[(.5, .5, -.5), (-.5, .5, -.5), (-.5, -.5, -.5), (.5, -.5, -.5)],
             tex=base(24, 8)),
    ]

    hat_coords = {
        "top": (40, 0), "bottom": (48, 0),
        "right": (48, 8), "front": (40, 8),
        "left": (32, 8), "back": (56, 8),
    }
    hat_faces = []
    has_hat = False
    scale = 1.12
    for f in faces:
        x0, y0 = hat_coords[f["name"]]
        tex = base(x0, y0)
        if any(t[3] > 10 for row in tex for t in row):
            has_hat = True
        hat_faces.append(dict(
            name=f["name"] + "_hat",
            normal=f["normal"],
            corners=[(cx * scale, cy * scale, cz * scale) for cx, cy, cz in f["corners"]],
            tex=tex,
        ))

    return faces, (hat_faces if has_hat else [])

def rotate_point(p, yaw, pitch):
    x, y, z = p
    cy, sy = math.cos(yaw), math.sin(yaw)
    x1 = x * cy + z * sy
    z1 = -x * sy + z * cy
    y1 = y
    cp, sp = math.cos(pitch), math.sin(pitch)
    y2 = y1 * cp - z1 * sp
    z2 = y1 * sp + z1 * cp
    x2 = x1
    return (x2, y2, z2)

def render_frame(faces, hat_faces, yaw, pitch, canvas_size, scale, bg_color, xoff, yoff):
    w = h = canvas_size
    canvas = [list(bg_color) for _ in range(w * h)]
    cx, cy = w / 2, h / 2

    def project(p3):
        x, y, z = p3

        x -= xoff
        y -= yoff

        return (cx + x * scale, cy - y * scale, z)

    def draw_face_list(face_list):
        items = []
        for f in face_list:
            rn = rotate_point(f["normal"], yaw, pitch)
            if rn[2] <= 0.001:
                continue
            rc = [rotate_point(c, yaw, pitch) for c in f["corners"]]
            pc = [project(c) for c in rc]
            avg_z = sum(c[2] for c in rc) / 4.0
            items.append((avg_z, pc, f["tex"]))
        items.sort(key=lambda it: it[0])

        for avg_z, pc, tex in items:
            tex_size = len(tex)
            p0 = pc[0]
            u_vec = (pc[1][0] - p0[0], pc[1][1] - p0[1])
            v_vec = (pc[3][0] - p0[0], pc[3][1] - p0[1])
            det = u_vec[0] * v_vec[1] - u_vec[1] * v_vec[0]
            if abs(det) < 1e-9:
                continue
            inv_det = 1.0 / det

            xs = [pt[0] for pt in pc]
            ys = [pt[1] for pt in pc]
            minx, maxx = max(0, int(min(xs)) - 1), min(w - 1, int(max(xs)) + 1)
            miny, maxy = max(0, int(min(ys)) - 1), min(h - 1, int(max(ys)) + 1)

            for py in range(miny, maxy + 1):
                for px in range(minx, maxx + 1):
                    dx = (px + 0.5) - p0[0]
                    dy = (py + 0.5) - p0[1]
                    u = (dx * v_vec[1] - dy * v_vec[0]) * inv_det
                    v = (u_vec[0] * dy - u_vec[1] * dx) * inv_det
                    if 0 <= u < 1 and 0 <= v < 1:
                        tx = min(tex_size - 1, int(u * tex_size))
                        ty = min(tex_size - 1, int(v * tex_size))
                        r, g, b, a = tex[ty][tx]
                        idx = py * w + px
                        inv_a = 255 - a
                        bg_r, bg_g, bg_b = canvas[idx]

                        canvas[idx] = [
                            (r * a + bg_r * inv_a) >> 8,
                            (g * a + bg_g * inv_a) >> 8,
                            (b * a + bg_b * inv_a) >> 8,
                        ]

    draw_face_list(faces)
    if hat_faces:
        draw_face_list(hat_faces)

    return canvas

def write(path, frames, width, height, delay_ms = 60, loop = True):
    pil_frames = [Image.new("RGB", (width, height)) for _ in frames]

    for img,frame in zip(pil_frames, frames):
        img.putdata([tuple(px) for px in frame])

    strip = Image.new("RGB", (width * len(pil_frames), height))
    for i, img in enumerate(pil_frames):
        strip.paste(img, (i * width, 0))

    strip = strip.quantize(
        colors = 256,
        method = Image.MEDIANCUT
    )

    qframes = [
        strip.crop((i * width, 0, (i + 1) * width, height))
        for i in range(len(pil_frames))
    ]

    qframes[0].save(
        path,
        save_all=True,
        append_images=qframes[1:],
        duration=delay_ms,
        loop=0 if loop else 1,
        optimize=True,
        disposal=2,
    )

def main():
    args = sys.argv[1:]
    if not args:
        print("Usage: python3 mc_head_spin.py <nickname> [output.gif] [--frames N] [--size PX]")
        sys.exit(1)

    username = args[0]
    output = "head_spin.gif"
    n_frames = 36
    canvas_size = 200

    xoff = 0.0
    yoff = 0.1
    pitch = -18

    rest = args[1:]
    i = 0
    while i < len(rest):
        a = rest[i]
        if a == "--output" and i + 1 < len(rest):
            output = rest[i + 1]; i += 2
        elif a == "--frames" and i + 1 < len(rest):
            n_frames = int(rest[i + 1]); i += 2
        elif a == "--size" and i + 1 < len(rest):
            canvas_size = int(rest[i + 1]); i += 2
        elif a == "--xoff" and i + 1 < len(rest):
            xoff = float(rest[i + 1]); i += 2
        elif a == "--yoff" and i + 1 < len(rest):
            yoff = float(rest[i + 1]); i += 2
        elif a == "--pitch" and i + 1 < len(rest):
            pitch = int(rest[i + 1]); i += 2
        else:
            i += 1

    png_data = fetch(username)

    print("Decoding PNG...")
    w, h, pixels = decode_png(png_data)

    print("Making head geometry...")
    faces, hat_faces = build_head_faces(pixels, w)

    scale = canvas_size * 0.34
    pitch = math.radians(pitch)
    bg_color = (54, 57, 63)

    frames = []
    print(f"Rendering {n_frames} frames...")
    for i in range(n_frames):
        yaw = 2 * math.pi * i / n_frames
        frame = render_frame(faces, hat_faces, yaw, pitch, canvas_size, scale, bg_color, xoff, yoff)
        frames.append(frame)

    print(f"Saving {output} ...")
    write(output, frames, canvas_size, canvas_size, delay_ms = 60, loop = True)
    print("Saved!")


if __name__ == "__main__":
    main()