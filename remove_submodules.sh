#!/bin/bash

echo "Removing Git submodules..."

# remove simple-knn
git submodule deinit -f submodules/simple-knn
git rm -f submodules/simple-knn
rm -rf .git/modules/submodules/simple-knn

# remove diff-gaussian-rasterization
git submodule deinit -f submodules/diff-gaussian-rasterization
git rm -f submodules/diff-gaussian-rasterization
rm -rf .git/modules/submodules/diff-gaussian-rasterization

# remove SIBR_viewers
git submodule deinit -f SIBR_viewers
git rm -f SIBR_viewers
rm -rf .git/modules/SIBR_viewers

# remove fused-ssim
git submodule deinit -f submodules/fused-ssim
git rm -f submodules/fused-ssim
rm -rf .git/modules/submodules/fused-ssim

# clean up git index and working tree
git commit -am "Remove submodules: simple-knn, diff-gaussian-rasterization, SIBR_viewers, fused-ssim"
git gc --prune=now
git config -f .git/config --remove-section submodule.submodules/simple-knn 2>/dev/null
git config -f .git/config --remove-section submodule.submodules/diff-gaussian-rasterization 2>/dev/null
git config -f .git/config --remove-section submodule.SIBR_viewers 2>/dev/null
git config -f .git/config --remove-section submodule.submodules/fused-ssim 2>/dev/null

echo "Done."
