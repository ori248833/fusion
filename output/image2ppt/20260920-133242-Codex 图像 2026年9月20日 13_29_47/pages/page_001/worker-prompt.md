Rebuild one page for Image2PPT.

Run dir: D:\github\fusion\output\image2ppt\20260920-133242-Codex 图像 2026年9月20日 13_29_47
Page id: page_001
Page dir: D:\github\fusion\output\image2ppt\20260920-133242-Codex 图像 2026年9月20日 13_29_47\pages\page_001
Source image: D:\github\fusion\output\image2ppt\20260920-133242-Codex 图像 2026年9月20日 13_29_47\pages\page_001\source.png

You own only this Page dir. Do not edit deck_manifest.json, page_jobs.json, notes_manifest.json, final outputs, the original input, or any other page directory.

MANDATORY FIRST ACTION — before looking at the source image, before any decision, before any tool call other than reading: read these three files in full. Do not skim, do not rely on prior knowledge of them, do not start reconstruction first and consult them later. Every past failure mode of this skill is encoded in them; any decision made without having read them is invalid and will be redone.
- C:\Users\Administrator\.codex\skills\nature-image2ppt/references/page-decision-tree.md — the single source of truth for all object-source decisions: the three-step decision process, text-hints usage, the final self-check, and the fix-versus-warning split.
- C:\Users\Administrator\.codex\skills\nature-image2ppt/references/manifest-schema.md — the field contracts for manifest.json, validation.json, page_result.json, and imagegen-jobs.json.
- C:\Users\Administrator\.codex\skills\nature-image2ppt/references/cli-helper.md — local Image2PPT command syntax and examples.

Hard rules (reminders only; the details and rationale live in the references above):
1. Every non-text foreground visual object must be separated through the image-edit asset-sheet workflow per page-decision-tree.md section 2. Backend fallback never permits an object-source fallback: no native-shape/emoji/text-symbol approximation, no direct source.png crops, no downgrade to a warning.
2. Execute the three steps in order: (1) background recognition and repair, (2) foreground asset separation, (3) native element reconstruction. Do not consume the text hints in your page dir before the step-1/2 decisions are recorded.
3. manifest.json is the authoritative build source for page validation and final deck assembly. Build page.pptx and preview.png from manifest.json with the deterministic runtime, never with separate page-local PowerPoint code that bypasses the manifest.
4. All box_px / points_px / polygon_px values are source.png pixels. Reuse page_request.json.slide and page_request.json.content_box unchanged — do not convert the page to 16:9 or recalculate the canvas; the runtime maps source-pixel coordinates into content_box. Positioned objects without coordinates are page failures.
5. validation.json must contain a top-level boolean `passed`. Deterministic validation passing never waives an object-source rule.
6. Every page path and write must resolve inside this Page dir. Do not use `..`, an absolute escape, or a symlink to reach another page, the run root, or any external directory.
7. New manifests use `schema_version: 2`, structured visual classifications, and specific `quality_evidence`; a formula render failure remains a hard failure unless the user explicitly approves that exact exception in the manifest.

Image backend: execute `page_request.json.image_backend` using the authoritative field contract in `manifest-schema.md`. For `backend_id: builtin-imagegen`, the high-risk reminder is: use `image_gen.imagegen` first; generation needs only `prompt`, while editing requires `view_image` first and then `prompt` plus absolute local `referenced_image_paths`. Missing `mask`, `model`, `size`, `quality`, or `out` never triggers fallback. Import only the exact valid local result path (`output_hint` when supplied), never a scanned "newest" file; enter `'C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' 'C:\Users\Administrator\.codex\skills\nature-image2ppt\cli\image2ppt\cli.py' image generate/edit` only for a matching `fallback_policy.on` event. If that fallback cannot produce the required image, stop the page with `validation.json.passed=false`. In a network-restricted runtime, request any required approval and state that only task-local prompts and required page images/masks/references are uploaded for this user-requested conversion.

Goal: rebuild the source page as object-level editable PowerPoint. Do not invent an object-source strategy outside `page-decision-tree.md`.

If the page dir already contains artifacts (manifest.json, page.pptx, validation.json, assets, ...) from a previous failed attempt, treat them as untrusted: run the full decision process yourself and re-derive every artifact. Never flip a leftover validation.json to `passed: true` or return leftover outputs without having rebuilt and re-verified them — the previous attempt failed for a reason recorded in its validation.json; read it.

Work through the page in this order:
1. Build the page inventory (Pre-Decision Checklist in page-decision-tree.md).
2. Decide the background (page-decision-tree.md section 1) and record `background_strategy`.
3. Decide and separate foreground assets (section 2). Run step-1/2 image jobs serially through the backend order above; do not use a batch interface. Put icons/foreground objects onto one sparse asset sheet when they fit, with generous gaps between objects for clean splitting; create multiple sheets only when one sheet cannot fit them. After each selected local output, record and process it with `'C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' 'C:\Users\Administrator\.codex\skills\nature-image2ppt\cli\image2ppt\cli.py' image import` and `'C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' 'C:\Users\Administrator\.codex\skills\nature-image2ppt\cli\image2ppt\cli.py' image process-sheet`.
4. Rebuild native text, shapes, and tables (section 3). Fill `text_boxes` from the measured text hints per section 3.1; render formulas with `'C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' 'C:\Users\Administrator\.codex\skills\nature-image2ppt\cli\image2ppt\cli.py' formula render-latex` per section 3.2.
5. Write a schema-v2 manifest.json following the field contracts in manifest-schema.md, including `text_inventory`, structured `visual_inventory`, `background_strategy`, `quality_checks`, `quality_evidence`, and positioned `text_boxes`/`images`/`shapes`.
6. Build the artifacts with the deterministic runtime: `'C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' 'C:\Users\Administrator\.codex\skills\nature-image2ppt\cli\image2ppt\cli.py' page build D:\github\fusion\output\image2ppt\20260920-133242-Codex 图像 2026年9月20日 13_29_47\pages\page_001` (writes page.pptx and preview.png from manifest.json), then `'C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' 'C:\Users\Administrator\.codex\skills\nature-image2ppt\cli\image2ppt\cli.py' page contact-sheet D:\github\fusion\output\image2ppt\20260920-133242-Codex 图像 2026年9月20日 13_29_47\pages\page_001`, then `'C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' 'C:\Users\Administrator\.codex\skills\nature-image2ppt\cli\image2ppt\cli.py' page validate D:\github\fusion\output\image2ppt\20260920-133242-Codex 图像 2026年9月20日 13_29_47\pages\page_001` — it runs the same manifest-contract checks `'C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' 'C:\Users\Administrator\.codex\skills\nature-image2ppt\cli\image2ppt\cli.py' run record` will run, so fix every reported issue here, inside the page.

The Page dir must contain when you return:
- manifest.json
- imagegen-jobs.json
- page.pptx
- preview.png
- split_assets_contact.png
- validation.json
- page_result.json

validation.json and page_result.json must follow the exact shapes defined in manifest-schema.md: validation.json carries the top-level boolean `passed` (not only a nested or renamed field), and page_result.json carries the minimal required key set.

Before returning, run the Final Self-Check in page-decision-tree.md once: compare preview.png and split_assets_contact.png to the source, confirm `'C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' 'C:\Users\Administrator\.codex\skills\nature-image2ppt\cli\image2ppt\cli.py' page validate D:\github\fusion\output\image2ppt\20260920-133242-Codex 图像 2026年9月20日 13_29_47\pages\page_001` passes, confirm validation.json contains top-level `passed: true`, and confirm all required outputs exist. Page-local issues are fixed inside the current page by you before returning.

On failure — when a hard rule cannot be satisfied or a required tool is unavailable — stop and return a page failure: write validation.json with `"passed": false` and the concrete failure reason (what failed, the exact error, what the parent must fix), plus page_result.json referencing whatever artifacts exist (omit keys for artifacts that were never produced). Do not fabricate the remaining artifacts and do not build an approximate page to make validation pass; the parent agent will fix the root cause and dispatch or claim a fresh page execution.

Return only:
page_manifest=`<absolute path>`
page_pptx=`<absolute path>`
preview=`<absolute path>`
contact_sheet=`<absolute path>`
validation=`<absolute path>`
page_result=`<absolute path>`

# Image2PPT profile addendum

This addendum extends the complete local page-reconstructor prompt above. It does not replace that prompt, its required reading, its object-source decisions, its OCR/text-hint behavior, its ownership boundary, or its standard artifacts.

Image2PPT root: C:\Users\Administrator\.codex\skills\nature-image2ppt
Run directory: D:\github\fusion\output\image2ppt\20260920-133242-Codex 图像 2026年9月20日 13_29_47
Page id: page_001
Page directory: D:\github\fusion\output\image2ppt\20260920-133242-Codex 图像 2026年9月20日 13_29_47\pages\page_001
Source image: D:\github\fusion\output\image2ppt\20260920-133242-Codex 图像 2026年9月20日 13_29_47\pages\page_001\source.png

Before authoring the page, read these profile files in full:

- `C:\Users\Administrator\.codex\skills\nature-image2ppt/references/region-decomposition.md`
- `C:\Users\Administrator\.codex\skills\nature-image2ppt/references/object-routing.md`
- `C:\Users\Administrator\.codex\skills\nature-image2ppt/references/manifest-arrow-extension.md`
- `C:\Users\Administrator\.codex\skills\nature-image2ppt/references/qa-contract.md`

Additional hard rules:

1. Keep `manifest.json` as the only page build source. Use schema version 2 with structured `visual_inventory` and `quality_evidence`. Add arrow fields directly to standard `shapes[]`; do not create an Image2PPT plan, OCR copy, review manifest, or controller state.
2. Before choosing individual objects, divide a structured page into 3-5 semantic regions (1-2 only for a genuinely simple page). Inventory and route each region with the region/mixed-reconstruction contract, then express the result only through standard manifest objects.
3. Add `image2ppt_region_decomposition` directly to `manifest.json`. For compound diagrams, record every node center/size and every edge endpoint/direction in source pixels, map each to a stable manifest id, and protect every node/edge as a visual anchor. This extension is evidence, not a second build plan.
4. Regular circles, nodes, cards, straight lines, dashed relationships, and ordinary connectors that can be measured accurately must remain native objects. Use bounded source-faithful/textless assets only for the complex local subparts that would visibly drift; never flatten a whole knowledge graph or compound region into one bitmap.
5. A simple arrow is exactly one manifest shape and one PowerPoint object. Use one line item with native start/end arrowhead fields, or one filled-arrow preset with optional embedded text. Never use a line plus triangle, Unicode arrow glyph, or grouped arrow fragments.
6. The local page-decision-tree remains authoritative for OCR ownership, text hints, background/image-edit provenance, formulas, and asset generation. Region routing changes page decomposition and mixed reconstruction, not those upstream ownership rules.
7. Stable unique ids are required for all `shapes[]`, `text_boxes[]`, and `images[]` referenced by a region. Connector captions outside the line remain ordinary `text_boxes[]`; centered labels inside a filled arrow use that same arrow shape's `text` field.
8. Every page artifact, manifest path, asset, formula, report, and output override must resolve inside `D:\github\fusion\output\image2ppt\20260920-133242-Codex 图像 2026年9月20日 13_29_47\pages\page_001`. A rejected path escape is a hard page failure.
9. Formula rendering failure is a hard page failure unless the user explicitly approved that exact exception and `formula_inventory` records `user_approved_exception: true` plus a concrete `approval_note`.

Replace only the base prompt's final page build/validate sequence with this extended sequence:

1. Finish the standard `manifest.json` and `imagegen-jobs.json` exactly as required by the base prompt, including `image2ppt_region_decomposition` inside the manifest.
2. Run `'C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' 'C:\Users\Administrator\.codex\skills\nature-image2ppt\cli\image2ppt\cli.py' page build D:\github\fusion\output\image2ppt\20260920-133242-Codex 图像 2026年9月20日 13_29_47\pages\page_001`.
3. Run `python C:\Users\Administrator\.codex\skills\nature-image2ppt/scripts/run_image2ppt_qa.py D:\github\fusion\output\image2ppt\20260920-133242-Codex 图像 2026年9月20日 13_29_47\pages\page_001`. A return code of 2 with only pending visual review is expected on the first pass.
4. Inspect `D:\github\fusion\output\image2ppt\20260920-133242-Codex 图像 2026年9月20日 13_29_47\pages\page_001\source.png` and `D:\github\fusion\output\image2ppt\20260920-133242-Codex 图像 2026年9月20日 13_29_47\pages\page_001/render/rendered.png` at useful detail, region by region. For compound diagrams check node count, center, size, circle geometry, edge endpoints, direction, dash style, document nodes, label anchors, and z-order. Also check every arrow's bend, head size, thickness, object count, and embedded label plus all base visual checks.
5. The first QA run writes `D:\github\fusion\output\image2ppt\20260920-133242-Codex 图像 2026年9月20日 13_29_47\pages\page_001/visual-review-evidence.template.json`. Fix `manifest.json` or assets, rebuild, and repeat until the render matches. Copy the current template to `visual-review-evidence.json`, set `reviewed` to true, and fill every required check with a specific source-versus-render observation. Then run:

   `python C:\Users\Administrator\.codex\skills\nature-image2ppt/scripts/run_image2ppt_qa.py D:\github\fusion\output\image2ppt\20260920-133242-Codex 图像 2026年9月20日 13_29_47\pages\page_001 --visual-review-status reviewed --visual-review-evidence D:\github\fusion\output\image2ppt\20260920-133242-Codex 图像 2026年9月20日 13_29_47\pages\page_001/visual-review-evidence.json`

   Generic notes such as "looks good" never satisfy this evidence contract. Supplemental `--visual-review-notes` may explain context but cannot replace the evidence file.
6. Run `'C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' 'C:\Users\Administrator\.codex\skills\nature-image2ppt\cli\image2ppt\cli.py' page contact-sheet D:\github\fusion\output\image2ppt\20260920-133242-Codex 图像 2026年9月20日 13_29_47\pages\page_001` after the accepted build. Confirm standard `validation.json` still contains top-level `passed: true` and its `image2ppt_profile.passed` is true.
7. Write the standard `page_result.json` shape required by the local manifest schema. Do not add an alternative result file.

In addition to the base-required files, leave these supplemental reports in the page directory:

- `arrow_postprocess_report.json`
- `arrow_inspection_report.json`
- `region_decomposition_report.json`
- `render/rendered.png`
- `render_report.json`
- `image2ppt_qa.json`
- `visual-review-evidence.json`

Return exactly the standard paths requested in the base prompt.
