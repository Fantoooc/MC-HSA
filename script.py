import sys, io, math, json, base64
import requests
import numpy as np
from PIL import Image

def fetch(username: str) -> bytes:
    headers = { "User-Agent": "minecraft-heads-spin-script/1.0"}
    try:
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

def decode(data: bytes):
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("Not a png file has been downloaded")

    with Image.open(io.BytesIO(data)) as img:
        img = img.convert("RGBA")
        width, height = img.size
        pixels = np.asarray(img)

    return width, height, pixels

def extract(pixels: np.ndarray, x0: int, y0: int, size: int = 8) -> np.ndarray:
    return pixels[y0 : y0 + size, x0 : x0 + size]

def build_faces(pixels: np.ndarray):
    # x: right
    # y: up
    # z: front
    tex_normals = np.array([
        (0, 1, 0),  # Top
        (0, -1, 0), # Bottom
        (1, 0, 0),  # Right
        (0, 0, 1),  # Front
        (-1, 0, 0), # Left
        (0, 0, -1)  # Back
    ], dtype = np.float32)

    base_off = [(8, 0), (16, 0), (16, 8), (8, 8), (0, 8), (24, 8)]
    hat_off  = [(40, 0), (48, 0), (48, 8), (40, 8), (32, 8), (56, 8)]

    corners = np.array([
        [(-.5, .5, -.5), (.5, .5, -.5), (.5, .5, .5), (-.5, .5, .5)],     # Top
        [(-.5, -.5, -.5), (.5, -.5, -.5), (.5, -.5, .5), (-.5, -.5, .5)], # Bottom
        [(.5, .5, .5), (.5, .5, -.5), (.5, -.5, -.5), (.5, -.5, .5)],     # Right
        [(-.5, .5, .5), (.5, .5, .5), (.5, -.5, .5), (-.5, -.5, .5)],     # Front
        [(-.5, .5, -.5), (-.5, .5, .5), (-.5, -.5, .5), (-.5, -.5, -.5)], # Left
        [(.5, .5, -.5), (-.5, .5, -.5), (-.5, -.5, -.5), (.5, -.5, -.5)]  # Back
    ], dtype = np.float32)

    tex     = np.stack([extract(pixels, x0, y0) for x0, y0 in base_off])
    hat_tex = np.stack([extract(pixels, x0, y0) for x0, y0 in hat_off])

    has_hat = (hat_tex[..., 3] > 10).any()
    scale = 1.12

    faces     = dict(normal=tex_normals, corners=corners, tex=tex)
    hat_faces = dict(normal=tex_normals, corners=corners * scale, tex=hat_tex) if has_hat else None

    return faces, hat_faces

def rotate_points(points, yaw, pitch):
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)

    x, y, z = points[..., 0], points[..., 1], points[..., 2]

    x1 = x * cy + z * sy
    z1 = -x * sy + z * cy
    y1 = y

    y2 = y1 * cp - z1 * sp
    z2 = y1 * sp + z1 * cp
    x2 = x1

    return np.stack([x2, y2, z2], axis = -1)

def render_frame(faces, hat_faces, yaw, pitch, canvas_size, scale, bg_color, xoff, yoff):
    w = h = canvas_size
    canvas = np.empty((w, h, len(bg_color)), dtype = np.float32)
    canvas[:] = bg_color
    cx, cy = w / 2, h / 2

    def project(points):
        x, y, z = points[..., 0], points[..., 1], points[..., 2]

        x -= xoff
        y -= yoff

        px = cx + x * scale
        py = cy - y * scale

        return np.stack([px, py, z], axis = -1)

    def draw_faces(faces):
        normals = faces["normal"]
        corners = faces["corners"]
        tex     = faces["tex"]

        rn = rotate_points(normals, yaw, pitch)
        rc = rotate_points(corners, yaw, pitch)
        pc = project(rc)

        visible = rn[..., 2] > 0.001
        avg_z   = rc[..., 2].mean(axis = 1)

        order = np.where(visible)[0]
        order = order[np.argsort(avg_z[order])]

        for i in order:
            face_pc  = pc[i]
            face_tex = tex[i]
            tex_size = face_tex.shape[0]

            p0 = face_pc[0, :2]
            u_vec = face_pc[1, :2] - p0
            v_vec = face_pc[3, :2] - p0
            det = u_vec[0] * v_vec[1] - u_vec[1] * v_vec[0]
            if abs(det) < 1e-9:
                continue
            inv_det = 1.0 / det

            xs, ys = face_pc[:, 0], face_pc[:, 1]
            minx = max(0, int(np.floor(xs.min())) - 1)
            maxx = min(w - 1, int(np.ceil(xs.max())) + 1)
            miny = max(0, int(np.floor(ys.min())) - 1)
            maxy = min(h - 1, int(np.ceil(ys.max())) + 1)
            if maxx < minx or maxy < miny:
                continue

            py_grid, px_grid = np.mgrid[miny:maxy + 1, minx:maxx + 1]
            dx = (px_grid + 0.5) - p0[0]
            dy = (py_grid + 0.5) - p0[1]
            u = (dx * v_vec[1] - dy * v_vec[0]) * inv_det
            v = (u_vec[0] * dy - u_vec[1] * dx) * inv_det

            mask = (u >= 0) & (u < 1) & (v >= 0) & (v < 1)
            if not mask.any():
                continue

            tx = np.clip((u * tex_size).astype(int), 0, tex_size - 1)
            ty = np.clip((v * tex_size).astype(int), 0, tex_size - 1)
            texel = face_tex[ty, tx]

            a = texel[..., 3:4].astype(np.float32) / 255.0
            rgb = texel[..., :3].astype(np.float32)

            region = canvas[miny:maxy + 1, minx:maxx + 1]
            blended = rgb * a + region * (1 - a)
            region[mask] = blended[mask]
            canvas[miny:maxy + 1, minx:maxx + 1] = region

    draw_faces(faces)
    if hat_faces:
        draw_faces(hat_faces)

    return np.clip(canvas, 0, 255).astype(np.uint8)

def write(path, frames, width, height, delay_ms = 60, loop = True):
    if frames is None or len(frames) == 0: return

    strip = Image.new("RGB", (width * len(frames), height))
    for i, frame in enumerate(frames):
        with Image.fromarray(frame) as img:
            strip.paste(img, (i * width, 0))

    strip = strip.quantize(
        colors = 256,
        method = Image.MEDIANCUT
    )

    qframes = [
        strip.crop((i * width, 0, (i + 1) * width, height))
        for i in range(len(frames))
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
    output = "output.gif"
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

    scale = canvas_size * 0.34
    pitch = math.radians(pitch)
    bg_color = (54, 57, 63)

    print(f"Parsing {username}'s minecraft skin")
    png_data = fetch(username)

    print("Decoding PNG...")
    img_w, img_h, pixels = decode(png_data)

    print("Making head geometry...")
    faces, hat_faces = build_faces(pixels)

    print(f"Rendering {n_frames} frames...")
    frames = np.array([
        render_frame(faces, hat_faces, 2 * math.pi * i / n_frames, pitch, canvas_size, scale, bg_color, xoff, yoff)
        for i in range(n_frames)
    ])
    print(f"Saving {output} ...")
    write(output, frames, canvas_size, canvas_size, delay_ms = 60, loop = True)
    print("Saved!")


if __name__ == "__main__":
    main()