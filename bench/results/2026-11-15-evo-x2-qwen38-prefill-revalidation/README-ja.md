# EVO / Qwen 128K 再検証報告

実施日：2026年11月15日（JST）
判定：**温度上限で途中停止。従来版への復元確認済み。本番採用は保留。**

## 得られた結果

同じ fork バイナリで、実装済みの native fused 経路を明示的に有効にした C1 は、無効の C0 より速かった。完了した2回の中央値では、P1 の prefill 時間が **9.61%短縮**、P2 の2要求完了時間が **8.65%短縮**した。

ただし、予定した3反復は完了していない。第3回 B0 の P2 開始直後に GPU 温度が停止基準へ達したため、試験を中止した。C1 の出力は C0 と一致せず、モデル品質の同等性も未確認である。この結果だけでは本番へ切り替えられない。

### 中央値

| 条件 | 完了数 P1 / P2 | P1 prefill 秒 | P1 TTFT 秒 | P1 prefill tok/s | P1 decode tok/s | P2 2要求完了 秒 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B0：upstream、fused 無効 | 3 / 2 | 241.724 | 241.831 | 540.1 | 27.1 | 488.505 |
| C0：fork、fused 無効 | 2 / 2 | 244.823 | 244.940 | 533.3 | 27.0 | 492.882 |
| C1：C0 と同じバイナリ、native fused 有効 | 2 / 2 | 221.289 | 221.394 | 590.0 | 27.0 | 450.244 |

C0 は B0 より prefill 時間が1.28%長かった。前回の公式版0.1.58と fork 0.1.59.1の比較にあった約4%差を、fork の変更だけに帰属することはできない。今回も反復数が少なく、実行順・温度・計測負荷があるため、母集団の性能差を証明したとは扱わない。

### 全完了値

| run | P1 prefill 秒 | P1 TTFT 秒 | prefill tok/s | decode tok/s | 出力 tokens | P2 2要求完了 秒 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| r1-b0 | 241.724 | 241.831 | 540.1 | 27.1 | 53 | 488.560 |
| r1-c0 | 245.281 | 245.395 | 532.3 | 27.0 | 53 | 493.512 |
| r1-c1 | 219.139 | 219.248 | 595.7 | 26.9 | 42 | 445.985 |
| r2-c1 | 223.439 | 223.540 | 584.3 | 27.1 | 42 | 454.503 |
| r2-c0 | 244.364 | 244.484 | 534.3 | 27.0 | 53 | 492.251 |
| r2-b0 | 241.800 | 241.915 | 539.9 | 26.9 | 53 | 488.450 |
| r3-b0 | 240.955 | 241.069 | 541.8 | 27.1 | 53 | 温度停止 |

第3回 B0 の P1 は正常完了したので残している。P2 は未完了で、中央値へ混ぜていない。第3回 C0、第3回 C1、公式版 A0 の参考測定は未実施。欠測を成功・ゼロ秒として扱っていない。

## 何を直したか

1. **評価構成の欠落を修正した。** Linux gfx1151 は Windows の fused 初期設定を引き継がないため、既存の native fused 経路を明示的に選ぶ実験用構成を追加した。C0/C1 の変更点は `STRATA_PF_FUSED=0` / `1` の1つ。`STRATA_PF_FUSED_NATIVE=1`、`STRATA_PF_F16=0`、`STRATA_PF_DM16=0` は共通とした。既存経路の有効化であり、新しい direct-FP16 カーネルの実装ではない。
2. **実行経路を確認できる診断を追加した。** qtype、gate/up と down の分岐、Dm 型、タイル行数を、起動エラー確認後に記録する。誤解を招く固定の int8 表示も修正した。診断は各型・分岐の最初の呼出しを記録するもので、全層の実行回数や GPU 完了そのものを保証しない。
3. **ベンチの絶対 deadline を修正した。** HTTP 接続、ヘッダー待ち、本文受信を含めて900秒を上限とし、接続が detached になった場合も socket を閉じて終了する。遅れて届いた本文・EOFを成功にしない。停止・回収の回帰テストを追加した。

実装されていない `direct64` / `pair128` を有効な最適化として表示していない。P2 のスケジューラ、既定設定、本番設定は変更していない。C1 の経路は upstream と共有する既存実装であり、fork 固有の新カーネルによる優位性を示した測定でもない。

## 測定条件

- AMD Radeon 8060S iGPU、gfx1151。RTX ではない。
- Qwen3.8-Flash-Next の UD-IQ3_XXS native pack。実際の gate/up・down は混合 qtype。モデルの既存 alias 名から純粋な IQ3_S と解釈しない。
- 入力は各要求130,552 tokens、context は131,072、server slots は2。P1 は同時要求1、P2 は同時要求2であり、P1 を「slots 1」と呼ばない。
- 同一の固定 prompt bytes、seed 42、temperature 0、top-k 1、top-p 1、thinking 無効、出力上限512。自然 EOS を尊重した。
- 各 arm は新しいプロセスで起動し、その中で P1、P2 の順に実施。全完了要求で native cache_n=0 を確認した。追加の短文性能試験は実施していない。
- B/C は同じビルドイメージ、Clang 22.0.0 / GCC 13.3.0、同じ CMakeCache、HIP gfx1151、engine 0.1.59.1。B/C の比較対象へ同じ診断パッチを適用した。
- 同じ server source、実行イメージ、モデル資産、hipBLASLt table、16 CPU / 104 GiB、candidate swap 上限0を使用した。既存の大容量重みは再ハッシュせず、同じ読み取り専用資産の size・inode・mtime と、小さい metadata/tokenizer の hash を照合した。
- prefill chunk 2048、expert cache 19000、resident budget 2、KV int8 / kv-resident 0（全量 GPU 常駐）、prompt cache 6、conv 2048 MiB / 4 slots、spec 4、adaptive 0、HC_Q8 0、route_tail_skip 0。
- `STRATA_PREFILL_TIMING=1`、`STRATA_HIPBLASLT_VERBOSE=1`、`STRATA_PF_FUSED_TRACE=1` を全 arm に揃えた**診断付き測定**。診断負荷がゼロとは主張しない。最初の2変数は `0` でも有効となるので、無効化には unset が必要。

## 実行経路・正しさの確認

以下の4組について、既存の `hip_prefill_fused_iq --edge --no-timing` の parity test が実機で成功した。

| Gate/up qtype | Down qtype | pack 内の層数 |
| --- | --- | ---: |
| IQ2_S (22) | IQ4_NL (20) | 23 |
| IQ2_S (22) | Q2_0 (42) | 3 |
| IQ3_XXS (18) | IQ4_NL (20) | 21 |
| IQ4_XS (23) | IQ4_NL (20) | 1 |

C1 の完了した両反復で、4組すべてに `gate_up=wmma-int8 down=wmma-int8 dm=fp32 activation_staging=int8` の実行ログが出た。FP16 の直接演算ではない。

hipBLASLt は gfx1151 / version 100401 用の90行の tuning table を読み込んだ。完了した6 arm それぞれの本計測側 summary は launches=131712、fallbacks=0、unique_fallback_shapes=0。別の warm-up 側には launches=0、fallbacks=2 の summary があり、これを本計測の fallback と混同しない。選択された solution ID と形状は evidence.json に保存した。

GPU timeline は純粋な kernel 実行時間ではない。特に P2 の待機や yield が含まれることがあり、fused 時は gate/up の表示範囲に down を含むため、表示名だけで個別 phase を比較しない。

### 出力の違い

- P1：B0/C0 は全完了回で53 tokens、同じ text hash。C1 は両回とも42 tokens、B0/C0 とは異なる hashだった。
- P2：B0 は各回50/44 tokens。C0 は51/46、50/44。C1 は46/44、51/46だった。並びは client request 配列の順で、server の処理順とは限らない。
- kernel parity はモデル品質の同等性を示さない。正答率、perplexity、長文保持、安全性の品質評価は今回未実施。出力長が違うため、end-to-end 時間だけで数学的な同等性を主張しない。

## P2：待ち時間と prefill を分ける

受付順2番目の要求は、admitted になってから reading が始まるまで待っていた。C1 でもこの待機は残った。

| 条件 / 回 | admitted → reading 秒 | reading → first token 秒 |
| --- | ---: | ---: |
| C0 / 1 | 242.994 | 247.305 |
| C0 / 2 | 246.034 | 243.203 |
| C1 / 1 | 221.658 | 221.233 |
| C1 / 2 | 225.699 | 225.748 |

これらは server event の区間であり、native prefill timer、client TTFT、decode とは別の値である。最初の要求の decode と次の prefill は重なったが、2本の prefill が並列に進んだとは扱わない。最初の利用者の TTFT を悪化させて overlap を作る変更も加えていない。

## cgroup と温度

完了した P1 全7件では、candidate の `nr_throttled` / `throttled_usec` の増分はともに0だった。完了した P2 の throttled_usec 増分は B0：252466 / 272873、C0：22249 / 279066、C1：45920 / 73752 μs。親 system.slice と root の throttled_usec 増分は全完了ケースで0だった。これだけを根拠に source code の原因は断定しない。各ケースの memory events、PSI と host VM の差分も evidence.json に残した。

2026-11-15 23:23:01 JST、第3回 B0 の P2 開始直後に停止基準側 GPU 温度が90.0°Cへ達した。隣接した GPU telemetry 読取りは92.0°C、CPU は83.75°C、空き RAM は43.82 GiBだった。異なる時点の2つの温度値を平均して上限未満とは判定していない。既定の停止基準どおり中止し、上限を緩めず再試行もしていない。C1 中ではなく B0 中の停止だが、それだけで C1 の熱安全性を証明できるわけではない。

## 復元と残課題

- 従来版の停止確認：22:01:43 JST。復元確認：23:23:30 JST。
- 試験用 container の不在を確認後、元の unit / image / config へ戻した。engine 0.1.58、context 131072、slots 2、復元確認用の最小 canary が成功した。
- 併存 OCR の container identity・image・health は変化していない。watchdog と試験 controller は終了し、試験用 container は残っていない。
- 夜間試験中に、新しい本番設定への切替、追加の停止、push はしていない。
- 続けるなら、まず熱条件と再試験計画を確認する。安全上限を変えず、欠けた反復、診断を外した性能再確認、出力品質評価を分けて行う。本番採用は別途承認が必要。

## 再現情報

- 所有者 fork：<https://github.com/circlemouth/Strata>
- 作業 branch：`evo-x2-prefill-revalidation-20261115`（評価用。本番 branch へは反映していない）
- 実装・評価構成：`316d0846f21e4bf7c77104d8cddb68613110cb6a`
- deadline 修正：`d9851ab545784ad435f982932d7adf288f764851`
- B upstream：`13eedf552417a6a791a7d13c6597e24300ee3a09`
- C fork：`1fe214570764e46eba0d4b60c794d83b89cffd62`
- B binary SHA-256：`cd523c4c0f0490dc616a3a0ed9e679018ffb44ebbe3a2d09ae2e90888f7d873b`
- C binary SHA-256：`6da7d4aa2cfef76fdf3b7310dbe8177f4535df20d5cd01dbe1b24a3058562f05`
- 共通 kernel test binary：`392f63a660d530cd36cedc1db9210627adcc9d2c86861592bf4fcb3165e67572`
- hipBLASLt table SHA-256：`7e9e4e81b2306cab0906936e1e836fdf7f2f4e0805d17ab7596ea398a0946ac9`
- criteria SHA-256：`9eaca6cbe167d10ffb090113e453a03126b0bf1de3a3839dcb62944c1c12d617`
- seal revision 2 SHA-256：`a16eef05f70f19c0e34cb1743566169e84c26a3df37a76f04b470738230cc680`

offline benchmark tests は復元後にも165件を再実行し、全件成功した。事前検証：EVO tooling 20件成功・任意の jsonschema 照合1件スキップ、recovery mock tests 26件成功、B/C の HIP build 成功。CUDA/SYCL build は未実施。

数値の原本は同梱 `evidence.json`。private host name、ネットワーク、prompt 本文、生ログ、認証情報は収録していない。追加の読取り専用診断で数値ログを保存したが、r1-b0 と中断直前 r3-b0 の solution/phase の追加抜粋は欠測。r2-b0 と完了 C0/C1 の抜粋は保存している。試験中に seal や実行構成は変更していない。終了後にも seal の196入力ファイルを再ハッシュし、全件の一致を確認した。
