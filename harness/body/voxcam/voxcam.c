// voxcam: a first-person Minecraft (1.11) camera without a browser or a GPU. It keeps a box of blocks around the
// bot and ray-casts each pixel through it (textured blocks with Minecraft's face shading, distance fog, players as
// Steve-shaped boxes), then encodes a JPEG. A 256x256 picture takes a few ms on a many-core CPU, rendered from the
// state at the moment it is asked for, so it is never an old frame.
//
//   voxcam <texture dir (textures/1.11.2)> <width> <height>
// stdin, one command per line:
//   R x0 y0 z0 sx sy sz file   load the box [x0, x0+sx) x [y0, y0+sy) x [z0, z0+sz) from a file of uint16 block
//                              states (id << 4 | meta, little endian), index ((y * sz) + z) * sx + x
//   B x y z state              one block changed
//   F ex ey ez yaw pitch n [x y z yaw]*n   render from the eye (mineflayer yaw/pitch, radians) with n players
//   V deg                      the vertical field of view for the next frames (default 70; e.g. 20 for a zoomed view)
//   Q                          quit
// stdout: for each F, a 4-byte little-endian length and the JPEG bytes.
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#define TS 16                // texture size
#define MAXTEX 128
#define FAR 64.0f            // how far the camera sees (blocks)
#define FOG0 40.0f           // fog starts
#define FOV 70.0f            // vertical field of view, Minecraft's default

typedef struct { uint8_t p[TS * TS * 4]; } Tex;
static Tex tex[MAXTEX];
static char texname[MAXTEX][64];
static int ntex = 0;
static const char *texdir;
static uint8_t skin[64 * 64 * 4];
static int have_skin = 0;

// blocks: per state (id << 4 | meta) the texture of the top, the sides and the bottom (-1: not drawn), a tint
typedef struct { int16_t top, side, bot; uint8_t tint, cutout; } Look;
static Look looks[4096];

static int Wd, Ht;
static float fov = FOV;       // vertical field of view, degrees (V command)
static int X0, Y0, Z0, SX, SY, SZ;
static uint16_t *vox = NULL;

static int load_tex(const char *name) {
  for (int i = 0; i < ntex; i++) if (!strcmp(texname[i], name)) return i;
  if (ntex >= MAXTEX) return -1;
  char path[512];
  snprintf(path, sizeof path, "%s/blocks/%s.png", texdir, name);
  int w, h, n;
  uint8_t *img = stbi_load(path, &w, &h, &n, 4);
  if (!img || w != TS) { if (img) stbi_image_free(img); fprintf(stderr, "voxcam: no texture %s\n", name); return -1; }
  memcpy(tex[ntex].p, img, TS * TS * 4);   // animated textures are frames stacked downwards: the first one
  stbi_image_free(img);
  snprintf(texname[ntex], 64, "%s", name);
  return ntex++;
}

enum { T_NONE = 0, T_GRASS = 1, T_FOLIAGE = 2, T_WATER = 3 };
static void look(int id, int meta, const char *top, const char *side, const char *bot, int tint, int cutout) {
  for (int m = 0; m < 16; m++) {
    if (meta >= 0 && m != meta) continue;
    Look *L = &looks[(id << 4) | m];
    L->top = top ? load_tex(top) : -1;
    L->side = side ? load_tex(side) : L->top;
    L->bot = bot ? load_tex(bot) : L->top;
    L->tint = tint;
    L->cutout = cutout;
  }
}
static void cube(int id, int meta, const char *t) { look(id, meta, t, t, t, T_NONE, 0); }

static const char *COLORS[16] = {"white", "orange", "magenta", "light_blue", "yellow", "lime", "pink", "gray",
                                 "silver", "cyan", "purple", "blue", "brown", "green", "red", "black"};

static void init_looks(void) {
  for (int i = 0; i < 4096; i++) looks[i].top = looks[i].side = looks[i].bot = -2;   // -2: unknown, grey
  for (int m = 0; m < 16; m++) looks[m].top = looks[m].side = looks[m].bot = -1;     // air
  const char *stones[7] = {"stone", "stone_granite", "stone_granite_smooth", "stone_diorite", "stone_diorite_smooth",
                           "stone_andesite", "stone_andesite_smooth"};
  for (int m = 0; m < 7; m++) cube(1, m, stones[m]);
  look(2, -1, "grass_top", "grass_side", "dirt", T_GRASS, 0);
  cube(3, -1, "dirt"); cube(4, -1, "cobblestone"); cube(5, -1, "planks_oak"); cube(7, -1, "bedrock");
  look(8, -1, "water_still", NULL, NULL, T_WATER, 0); look(9, -1, "water_still", NULL, NULL, T_WATER, 0);
  cube(10, -1, "lava_still"); cube(11, -1, "lava_still"); cube(12, -1, "sand"); cube(13, -1, "gravel");
  for (int m = 0; m < 16; m++) {
    int birch = (m & 3) == 2;
    look(17, m, birch ? "log_birch_top" : "log_oak_top", birch ? "log_birch" : "log_oak", NULL, T_NONE, 0);
    look(18, m, birch ? "leaves_birch" : "leaves_oak", NULL, NULL, T_FOLIAGE, 1);
  }
  look(20, -1, "glass", NULL, NULL, T_NONE, 1);
  look(24, 0, "sandstone_top", "sandstone_normal", "sandstone_bottom", T_NONE, 0);
  look(24, 1, "sandstone_top", "sandstone_carved", "sandstone_bottom", T_NONE, 0);
  look(24, 2, "sandstone_top", "sandstone_smooth", "sandstone_top", T_NONE, 0);
  char name[64];
  for (int m = 0; m < 16; m++) {
    snprintf(name, sizeof name, "wool_colored_%s", COLORS[m]); cube(35, m, name);
    snprintf(name, sizeof name, "hardened_clay_stained_%s", COLORS[m]); cube(159, m, name);
  }
  int invisible[] = {31, 32, 37, 38, 83, 166, 175};   // grass, flowers, cane, barrier: not drawn
  for (unsigned i = 0; i < sizeof invisible / sizeof *invisible; i++) look(invisible[i], -1, NULL, NULL, NULL, 0, 0);
  look(44, -1, "stone_slab_top", "stone_slab_side", NULL, T_NONE, 0);
  cube(48, -1, "cobblestone_mossy"); cube(49, -1, "obsidian");
  look(51, -1, "fire_layer_0", NULL, NULL, T_NONE, 1);
  const char *bricks[4] = {"stonebrick", "stonebrick_mossy", "stonebrick_cracked", "stonebrick_carved"};
  for (int m = 0; m < 4; m++) cube(98, m, bricks[m]);
  cube(139, 0, "cobblestone"); cube(139, 1, "cobblestone_mossy");
  cube(169, -1, "sea_lantern");
  look(170, -1, "hay_block_top", "hay_block_side", NULL, T_NONE, 0);
  cube(172, -1, "hardened_clay");
}

static inline uint16_t at(int x, int y, int z) {
  x -= X0; y -= Y0; z -= Z0;
  if (x < 0 || y < 0 || z < 0 || x >= SX || y >= SY || z >= SZ) return 0;
  return vox[((size_t)y * SZ + z) * SX + x];
}

// ---------------------------------------------------------------- players: Steve-shaped boxes, textured with the skin

typedef struct { float x0, y0, z0, x1, y1, z1; int u, v, w, h, d; } Part;   // box in 1/16 blocks, skin origin and size
static const Part PARTS[] = {
  {-4, 24, -4, 4, 32, 4, 0, 0, 8, 8, 8},        // head
  {-4, 12, -2, 4, 24, 2, 16, 16, 8, 12, 4},     // body
  {-8, 12, -2, -4, 24, 2, 40, 16, 4, 12, 4},    // right arm
  {4, 12, -2, 8, 24, 2, 40, 16, 4, 12, 4},      // left arm
  {-4, 0, -2, 0, 12, 2, 0, 16, 4, 12, 4},       // right leg
  {0, 0, -2, 4, 12, 2, 0, 16, 4, 12, 4},        // left leg
};
#define SCALE (1.8f / 32.0f)   // the model is 32 pixels tall; a player 1.8 blocks

// skin pixel of a part's face at (s, t) in [0,1): the usual Minecraft skin layout (a box unfolded around u, v)
static void skin_rgb(const Part *P, int face, float s, float t, float *rgb) {
  int u0, v0, fw, fh;
  switch (face) {
    case 0: u0 = P->u + P->d; v0 = P->v + P->d; fw = P->w; fh = P->h; break;                 // front
    case 1: u0 = P->u + 2 * P->d + P->w; v0 = P->v + P->d; fw = P->w; fh = P->h; break;      // back
    case 2: u0 = P->u; v0 = P->v + P->d; fw = P->d; fh = P->h; break;                         // right side
    case 3: u0 = P->u + P->d + P->w; v0 = P->v + P->d; fw = P->d; fh = P->h; break;           // left side
    case 4: u0 = P->u + P->d; v0 = P->v; fw = P->w; fh = P->d; break;                         // top
    default: u0 = P->u + P->d + P->w; v0 = P->v; fw = P->w; fh = P->d; break;                // bottom
  }
  int px = u0 + (int)(s * fw), py = v0 + (int)(t * fh);
  if (!have_skin) { rgb[0] = 0.2f; rgb[1] = 0.5f; rgb[2] = 0.7f; return; }
  const uint8_t *c = &skin[(py * 64 + px) * 4];
  rgb[0] = c[0] / 255.f; rgb[1] = c[1] / 255.f; rgb[2] = c[2] / 255.f;
}

typedef struct { float x, y, z, yaw; } Player;

// nearest hit of a ray with a player's parts; returns distance (or FAR) and the colour
static float hit_player(const Player *pl, const float *o, const float *d, float *rgb) {
  // into the player's frame: feet at the origin, facing -z (mineflayer yaw 0 looks north, -z)
  float c = cosf(pl->yaw), s = sinf(pl->yaw);
  float ox = o[0] - pl->x, oy = o[1] - pl->y, oz = o[2] - pl->z;
  float lo[3] = {c * ox - s * oz, oy, s * ox + c * oz}, ld[3] = {c * d[0] - s * d[2], d[1], s * d[0] + c * d[2]};
  float best = FAR;
  for (unsigned k = 0; k < sizeof PARTS / sizeof *PARTS; k++) {
    const Part *P = &PARTS[k];
    float bmin[3] = {P->x0 * SCALE, P->y0 * SCALE, P->z0 * SCALE}, bmax[3] = {P->x1 * SCALE, P->y1 * SCALE, P->z1 * SCALE};
    float tn = -1e9f, tf = 1e9f; int axis = 0, neg = 0;
    for (int a = 0; a < 3; a++) {
      if (fabsf(ld[a]) < 1e-8f) { if (lo[a] < bmin[a] || lo[a] > bmax[a]) { tn = 1e9f; break; } continue; }
      float t1 = (bmin[a] - lo[a]) / ld[a], t2 = (bmax[a] - lo[a]) / ld[a];
      int n = 1;
      if (t1 > t2) { float tt = t1; t1 = t2; t2 = tt; n = 0; }
      if (t1 > tn) { tn = t1; axis = a; neg = n; }
      if (t2 < tf) tf = t2;
    }
    if (tn > tf || tn < 0 || tn >= best) continue;
    float h[3] = {lo[0] + ld[0] * tn, lo[1] + ld[1] * tn, lo[2] + ld[2] * tn};
    float fx = (h[0] - bmin[0]) / (bmax[0] - bmin[0]), fy = (h[1] - bmin[1]) / (bmax[1] - bmin[1]),
          fz = (h[2] - bmin[2]) / (bmax[2] - bmin[2]);
    int face; float ss, tt;
    if (axis == 2) { face = neg ? 0 : 1; ss = neg ? fx : 1 - fx; tt = 1 - fy; }        // front faces -z
    else if (axis == 0) { face = neg ? 2 : 3; ss = neg ? 1 - fz : fz; tt = 1 - fy; }
    else { face = neg ? 5 : 4; ss = fx; tt = fz; }
    if (ss < 0) ss = 0; if (ss > 0.999f) ss = 0.999f; if (tt < 0) tt = 0; if (tt > 0.999f) tt = 0.999f;
    skin_rgb(P, face, ss, tt, rgb);
    float shade = axis == 1 ? (neg ? 0.5f : 1.0f) : axis == 2 ? 0.8f : 0.6f;
    rgb[0] *= shade; rgb[1] *= shade; rgb[2] *= shade;
    best = tn;
  }
  return best;
}

// ---------------------------------------------------------------- blocks

static const float SKY[3] = {0.47f, 0.65f, 1.0f};
static void tint_rgb(int tint, float *rgb) {
  static const float T[4][3] = {{1, 1, 1}, {0.49f, 0.74f, 0.35f}, {0.47f, 0.67f, 0.18f}, {0.25f, 0.46f, 0.89f}};
  rgb[0] *= T[tint][0]; rgb[1] *= T[tint][1]; rgb[2] *= T[tint][2];
}

// march the grid (Amanatides-Woo); returns distance to the first drawn texel, its colour in rgb
static float march(const float *o, const float *d, float *rgb) {
  int x = (int)floorf(o[0]), y = (int)floorf(o[1]), z = (int)floorf(o[2]);
  int sx = d[0] > 0 ? 1 : -1, sy = d[1] > 0 ? 1 : -1, sz = d[2] > 0 ? 1 : -1;
  float dx = fabsf(1.f / (d[0] ? d[0] : 1e-9f)), dy = fabsf(1.f / (d[1] ? d[1] : 1e-9f)), dz = fabsf(1.f / (d[2] ? d[2] : 1e-9f));
  float tx = (sx > 0 ? (x + 1 - o[0]) : (o[0] - x)) * dx, ty = (sy > 0 ? (y + 1 - o[1]) : (o[1] - y)) * dy,
        tz = (sz > 0 ? (z + 1 - o[2]) : (o[2] - z)) * dz;
  float t = 0; int axis = -1;
  while (t < FAR) {
    uint16_t st = at(x, y, z);
    if (st && axis >= 0) {
      const Look *L = &looks[st];
      int ti = axis == 1 ? (sy > 0 ? L->bot : L->top) : L->side;
      if (ti != -1) {
        float hx = o[0] + d[0] * t, hy = o[1] + d[1] * t, hz = o[2] + d[2] * t, u, v;
        if (axis == 0) { u = hz - floorf(hz); v = 1 - (hy - floorf(hy)); if (sx < 0) u = 1 - u; }
        else if (axis == 2) { u = hx - floorf(hx); v = 1 - (hy - floorf(hy)); if (sz > 0) u = 1 - u; }
        else { u = hx - floorf(hx); v = hz - floorf(hz); }
        if (ti == -2) { rgb[0] = rgb[1] = rgb[2] = 0.55f; }
        else {
          int px = (int)(u * TS) & (TS - 1), py = (int)(v * TS) & (TS - 1);
          const uint8_t *c = &tex[ti].p[(py * TS + px) * 4];
          if (L->cutout && c[3] < 128) goto next;   // see-through texel (leaves, glass, fire)
          rgb[0] = c[0] / 255.f; rgb[1] = c[1] / 255.f; rgb[2] = c[2] / 255.f;
          int tint = L->tint;
          if (tint == T_GRASS && axis != 1) tint = T_NONE;   // grass sides carry their own colour
          if (tint) tint_rgb(tint, rgb);
        }
        float shade = axis == 1 ? (sy > 0 ? 0.5f : 1.0f) : axis == 2 ? 0.8f : 0.6f;
        rgb[0] *= shade; rgb[1] *= shade; rgb[2] *= shade;
        return t;
      }
    }
  next:
    if (tx < ty && tx < tz) { x += sx; t = tx; tx += dx; axis = 0; }
    else if (ty < tz) { y += sy; t = ty; ty += dy; axis = 1; }
    else { z += sz; t = tz; tz += dz; axis = 2; }
  }
  return FAR;
}

// ---------------------------------------------------------------- a frame

static uint8_t *img = NULL;
static uint8_t *jpg = NULL;
static size_t jpglen = 0, jpgcap = 0;
static void jpg_write(void *ctx, void *data, int size) {
  (void)ctx;
  if (jpglen + size > jpgcap) { jpgcap = (jpglen + size) * 2; jpg = realloc(jpg, jpgcap); }
  memcpy(jpg + jpglen, data, size);
  jpglen += size;
}

static void render(const float *eye, float yaw, float pitch, const Player *pl, int np) {
  float fwd[3] = {-sinf(yaw) * cosf(pitch), sinf(pitch), -cosf(yaw) * cosf(pitch)};
  float right[3] = {cosf(yaw), 0, -sinf(yaw)};
  float up[3] = {right[1] * fwd[2] - right[2] * fwd[1], right[2] * fwd[0] - right[0] * fwd[2], right[0] * fwd[1] - right[1] * fwd[0]};
  float th = tanf(fov * 0.5f * (float)M_PI / 180.f), aspect = (float)Wd / Ht;
#pragma omp parallel for schedule(dynamic, 8)
  for (int j = 0; j < Ht; j++) {
    for (int i = 0; i < Wd; i++) {
      float a = ((i + 0.5f) / Wd * 2 - 1) * th * aspect, b = (1 - (j + 0.5f) / Ht * 2) * th;
      float d[3] = {fwd[0] + a * right[0] + b * up[0], fwd[1] + a * right[1] + b * up[1], fwd[2] + a * right[2] + b * up[2]};
      float n = sqrtf(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
      d[0] /= n; d[1] /= n; d[2] /= n;
      float rgb[3], prgb[3];
      float t = march(eye, d, rgb);
      for (int k = 0; k < np; k++) {
        float tp = hit_player(&pl[k], eye, d, prgb);
        if (tp < t) { t = tp; rgb[0] = prgb[0]; rgb[1] = prgb[1]; rgb[2] = prgb[2]; }
      }
      if (t >= FAR) { float g = 0.85f + 0.15f * d[1]; rgb[0] = SKY[0] * g; rgb[1] = SKY[1] * g; rgb[2] = SKY[2] * g; }
      else if (t > FOG0) {
        float f = (t - FOG0) / (FAR - FOG0);
        for (int c = 0; c < 3; c++) rgb[c] = rgb[c] * (1 - f) + SKY[c] * f;
      }
      uint8_t *p = &img[(j * Wd + i) * 3];
      for (int c = 0; c < 3; c++) { float v = rgb[c] * 255.f; p[c] = v > 255 ? 255 : v < 0 ? 0 : (uint8_t)v; }
    }
  }
  // the crosshair, inverted against what is behind it, with a gap in the middle so that a far player right on it
  // (a few pixels tall) is not hidden
  int cx = Wd / 2, cy = Ht / 2, gap = Wd / 64 > 2 ? Wd / 64 : 2, r = gap + (Wd / 50 > 3 ? Wd / 50 : 3);
  for (int k = -r; k <= r; k++) {
    if (k >= -gap && k <= gap) continue;
    int pts[2][2] = {{cx + k, cy}, {cx, cy + k}};
    for (int q = 0; q < 2; q++) {
      uint8_t *p = &img[(pts[q][1] * Wd + pts[q][0]) * 3];
      p[0] = 255 - p[0]; p[1] = 255 - p[1]; p[2] = 255 - p[2];
    }
  }
  jpglen = 0;
  stbi_write_jpg_to_func(jpg_write, NULL, Wd, Ht, 3, img, 75);
}

int main(int argc, char **argv) {
  if (argc < 4) { fprintf(stderr, "usage: voxcam <texture dir> <width> <height>\n"); return 2; }
  texdir = argv[1]; Wd = atoi(argv[2]); Ht = atoi(argv[3]);
  img = malloc((size_t)Wd * Ht * 3);
  init_looks();
  char path[512];
  snprintf(path, sizeof path, "%s/entity/steve.png", texdir);
  int w, h, n;
  uint8_t *s = stbi_load(path, &w, &h, &n, 4);
  if (s && w == 64 && h >= 32) { memcpy(skin, s, 64 * (h > 64 ? 64 : h) * 4); have_skin = 1; }
  if (s) stbi_image_free(s);
  char line[65536];
  while (fgets(line, sizeof line, stdin)) {
    if (line[0] == 'Q') break;
    if (line[0] == 'R') {
      char file[1024];
      if (sscanf(line + 1, "%d %d %d %d %d %d %1023s", &X0, &Y0, &Z0, &SX, &SY, &SZ, file) != 7) continue;
      free(vox);
      size_t nv = (size_t)SX * SY * SZ;
      vox = calloc(nv, 2);
      FILE *f = fopen(file, "rb");
      if (f) { if (fread(vox, 2, nv, f) != nv) fprintf(stderr, "voxcam: short region file\n"); fclose(f); }
    } else if (line[0] == 'V') {
      float f;
      if (sscanf(line + 1, "%f", &f) == 1 && f > 1 && f < 170) fov = f;
    } else if (line[0] == 'B') {
      int x, y, z, st;
      if (sscanf(line + 1, "%d %d %d %d", &x, &y, &z, &st) == 4 && vox) {
        x -= X0; y -= Y0; z -= Z0;
        if (x >= 0 && y >= 0 && z >= 0 && x < SX && y < SY && z < SZ) vox[((size_t)y * SZ + z) * SX + x] = (uint16_t)st;
      }
    } else if (line[0] == 'F') {
      float eye[3], yaw, pitch; int np, off;
      Player pl[32];
      char *p = line + 1;
      if (sscanf(p, "%f %f %f %f %f %d%n", &eye[0], &eye[1], &eye[2], &yaw, &pitch, &np, &off) != 6) continue;
      p += off;
      if (np > 32) np = 32;
      for (int k = 0; k < np; k++) {
        if (sscanf(p, "%f %f %f %f%n", &pl[k].x, &pl[k].y, &pl[k].z, &pl[k].yaw, &off) != 4) { np = k; break; }
        p += off;
      }
      if (!vox) { SX = SY = SZ = 1; vox = calloc(1, 2); }
      render(eye, yaw, pitch, pl, np);
      uint32_t len = (uint32_t)jpglen;
      fwrite(&len, 4, 1, stdout);
      fwrite(jpg, 1, jpglen, stdout);
      fflush(stdout);
    }
  }
  return 0;
}
