# Test images

Ad-hoc images for eyeballing the pipeline. The `wicht_*` files come from the
Wicht & Hennebert Sudoku dataset (https://github.com/wichtounet/sudoku_dataset,
CC-BY-4.0); the `halfmixed` ones additionally have MNIST handwriting pasted in by
`scripts/make_mixed_sudoku.py`. The full dataset lives in `data/wicht_sudoku/`.

Each `wicht_*.jpg` has a `.dat` ground truth (9x9 grid, `0` = empty). The
`halfmixed` files also carry a `.json` listing which cells are printed and which
are handwritten.

| File | Source | Digits | Filled |
|---|---|---|---|
| `wicht_halfmixed_iphone_1005` | iPhone 5s, 960x1280 | printed + MNIST | ~65% |
| `wicht_halfmixed_galaxy_1072` | Galaxy S4, 960x1280 | printed + MNIST | ~65% |
| `wicht_halfmixed_lowres_140` | sonyEricsson, 640x480 | printed + MNIST | ~65% |
| `wicht_realpen_halffilled_1002` | iPhone 5s, real pen | printed + handwritten | partial |
| `wicht_realpen_complete_2002` | iPhone 5s, real pen | printed + handwritten | all 81 |
| `wicht_printed_iphone_1024` | iPhone 5s | printed only | clues only |
| `wicht_printed_lowres_32` | sonyEricsson, 640x480 | printed only | clues only |
| `wicht_mnist_allfilled_1009` | iPhone 5s | printed + MNIST | all 81 |

Citation: Wicht, B. and Hennebert, J., "Mixed handwritten and printed digit
recognition in Sudoku with Convolutional Deep Belief Network", ICDAR 2015.
