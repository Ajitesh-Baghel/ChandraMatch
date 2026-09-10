# Collects git state + the exact files Claude needs to inspect before
# integrating the v2 coarse structural engine, per CLAUDE_CHANDRAMATCH_HANDOFF.md
# Steps 1-3. Read-only. Does not download, install, or modify anything.
#
# Run in PowerShell from anywhere; it cd's into the repo itself.
#
# Output: E:\ChandraMatch_v2_context.txt  -> upload this file back to Claude.

$Root = "E:\SIH26166"
$Out  = "E:\ChandraMatch_v2_context.txt"

Set-Location $Root

function Write-Section($title) {
    Add-Content -Path $Out -Value ""
    Add-Content -Path $Out -Value ("=" * 100)
    Add-Content -Path $Out -Value $title
    Add-Content -Path $Out -Value ("=" * 100)
}

function Dump-File($path) {
    Write-Section "FILE: $path"
    if (Test-Path $path) {
        Get-Content -Path $path -Raw | Add-Content -Path $Out
    } else {
        Add-Content -Path $Out -Value "*** FILE NOT FOUND: $path ***"
    }
}

# Fresh output file
if (Test-Path $Out) { Remove-Item $Out }
New-Item -Path $Out -ItemType File | Out-Null

# --- Step 1: git state ---
Write-Section "GIT STATUS"
git status | Add-Content -Path $Out

Write-Section "GIT LOG (last 20)"
git log --oneline -20 | Add-Content -Path $Out

Write-Section "GIT LS-FILES"
git ls-files | Add-Content -Path $Out

Write-Section "LOCAL-ONLY / UNTRACKED FILES (git status --porcelain, untracked only)"
git status --porcelain | Where-Object { $_ -match '^\?\?' } | Add-Content -Path $Out

# --- Step 2: key docs and scripts ---
$FilesToRead = @(
    "README.md",
    "docs\AGENT_STATE.md",
    ".github\workflows\runner-health.yml",

    "scripts\benchmark_pair004_global.py",
    "scripts\benchmark_pair006_global.py",
    "scripts\benchmark_pair007_global.py",

    "scripts\auto_select_pair004_coarse_hypothesis.py",
    "scripts\auto_select_pair006_coarse_hypothesis.py",
    "scripts\auto_select_pair007_coarse_hypothesis.py",

    "scripts\pair006_tiled_lightglue.py",
    "scripts\pair007_tiled_lightglue.py",

    "scripts\validate_pair004_tiled_lightglue_perturbation.py",
    "scripts\validate_pair005_fine_perturbation.py",
    "scripts\validate_pair006_large_offset_perturbation.py",
    "scripts\validate_pair007_selected_perturbation.py",
    "scripts\validate_pair007_loftr_intensity_perturbation.py",

    "scripts\diagnose_pair007_large_offset.py",

    "src\geometry\coverage.py",
    "src\geometry\ransac.py",
    "src\geometry\sanity.py",

    "src\matching\rift2\__init__.py",
    "src\matching\rift2\feature.py",
    "src\matching\rift2\matcher.py",
    "src\matching\rift2\simple_matcher.py",
    "src\matching\sift.py",
    "src\matching\uniform.py",
    "src\matching\verify.py",

    "src\overlap\footprints.py",
    "src\overlap\warp_loc_to_tmc.py",
    "src\overlap\build_pair_004.py",
    "src\overlap\build_pair_006.py",
    "src\overlap\build_pair_007.py",
    "src\overlap\analyze_pair_006.py",
    "src\overlap\analyze_pair_007.py",

    "src\canonicalization\common_grid.py",
    "src\canonicalization\build_real_pair.py",

    "src\preprocessing\gradient.py",
    "src\preprocessing\lunar_preprocess.py",

    "src\refinement\local_phase.py",
    "src\registration\warp.py",
    "src\utils\io.py"
)

foreach ($f in $FilesToRead) {
    Dump-File $f
}

Write-Host ""
Write-Host "Done. Upload this file back to Claude:"
Write-Host $Out
