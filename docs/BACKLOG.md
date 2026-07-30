# Backlog

## Polyscope `ps.pick()` によるマウスグラブ対象選択の堅牢化

- **背景**: `examples/bags_collision.py` の `handle_direct_mouse_drag` は、掴む頂点を
  「マウスレイに最も近い頂点」を自前で線形探索して選んでいる（`ray_o + t_proj * ray_d` と
  全頂点の距離を比較）。この方式は奥行き（オクルージョン）を考慮しないため、手前の面に
  隠れている頂点でも距離が近ければ誤って掴んでしまう可能性がある。
- **改善案**: Polyscopeが提供するGPUベースのピッキングAPI `ps.pick(screen_coords=(x, y))`
  を使う。戻り値の `PickResult` には `is_hit` / `position`（ワールド座標のヒット点）/
  `local_index`（構造体内の頂点インデックス）/ `structure_name` が含まれ、実際に画面上に
  見えている表面だけを正確に拾える。
- **適用範囲の注意**: 置き換えられるのは「初期グラブ対象の選択」部分のみ。掴んだ後に
  マウスへ追従させて3D空間を動かす部分（何もない空間上の一点を動かす処理）は pick では
  拾えないため、現行のレイ+固定深度計算（`ray_o + ray_d * grabbed_depth`）はそのまま必要。
- **優先度**: 低〜中。現状の実装（2026-07-28/29 修正後）でも実用上は動作しているため、
  機能追加ではなく堅牢化・精度向上が目的。
- **参照**: `ps.pick()` / `ps.PickResult`（polyscope 2.6.1 で確認済み、`is_hit`,
  `position`, `depth`, `local_index`, `structure_name`, `structure_type_name` を保持）。
