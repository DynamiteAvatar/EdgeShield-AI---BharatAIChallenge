# EdgeShield AI — Phase 1 UI patch

This patch is built from the uploaded EdgeShield_UI_Source.zip and retains the existing SVM, Llama, and callback logic. It does NOT contain models, tokenizer, or SVM JSON assets.

## Install (Windows / Android Studio)
1. BACK UP the working EdgeShieldExecuTorch project first (or commit to Git).
2. Open this ZIP and copy the contents of `app/src/main/` into the same path in your project. Choose Merge folders / Replace `MainActivity.java` when prompted.
3. Do NOT delete your existing `app/src/main/assets` folder. It contains required SVM assets and the model folder.
4. Do NOT change build.gradle, Gradle, EdgeShieldSvm.java, or model/tokenizer files.
5. Android Studio > Build > Make Project, then run on POCO F7.
6. Tap LOAD LLAMA ENGINE, wait for both models to load, paste an SMS, then ANALYSE MESSAGE.

## Included
- `MainActivity.java` — same verified SVM/ExecuTorch inference logic; onCreate replaced with XML binding and paste/clear/character counter.
- `activity_main.xml` — light neo-noir home, input, local engine, result, credits.
- `edge_*.xml` drawables — vector shield, gradients, cards, buttons, translucent surface styling.

## Technical honesty
- This is the FIRST functional UI pass, not the final polished motion/real background blur implementation. XML semi-transparent panels approximate glass. Real blur requires further Android-specific implementation and device testing.
- This patch has been structurally checked but has NOT been compiled in the user's Android Studio environment.
- The known Hindi scam false negative remains; the patch intentionally does not alter the model.
- `LOAD LLAMA ENGINE` remains manual, preserving the existing stable loading path.
