import gzip
from pathlib import Path

bb_map = Path("ldetect_example/ref/chr2.interpolated_genetic_map.gz")
joe_map = Path("ldetect_original/data/maps/chr2.interpolated_genetic_map.gz")


def read_map(p):
    with gzip.open(p, "rt") as f:
        rows = [line.split() for line in f]
    positions = {int(r[1]): float(r[2]) for r in rows}
    return positions


bb = read_map(bb_map)
joe = read_map(joe_map)

bb_pos = set(bb)
joe_pos = set(joe)

print(f"BitBucket entries:   {len(bb_pos):,}")
print(f"joepickrell entries: {len(joe_pos):,}")
print(f"Only in BitBucket:   {len(bb_pos - joe_pos):,}")
print(f"Only in joepickrell: {len(joe_pos - bb_pos):,}")
print(f"In both:             {len(bb_pos & joe_pos):,}")

# Check genetic position agreement at shared sites
shared = bb_pos & joe_pos
diffs = [(p, bb[p], joe[p]) for p in list(shared)[:100] if abs(bb[p] - joe[p]) > 1e-6]
print(f"\nGenetic position mismatches (first 100 shared): {len(diffs)}")
if diffs:
    for p, b, j in diffs[:5]:
        print(f"  pos={p}  bb={b:.6f}  joe={j:.6f}")
