package com.edgeshield.executorch;

import android.os.Bundle;
import android.content.ClipboardManager;
import android.content.ClipData;
import android.content.Context;
import android.graphics.Color;
import android.text.Editable;
import android.text.TextWatcher;
import android.view.View;
import android.widget.Toast;
import android.util.Log;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;

import org.pytorch.executorch.extension.llm.LlmModule;
import org.pytorch.executorch.extension.llm.LlmCallback;
import org.pytorch.executorch.extension.llm.LlmGenerationConfig;

import java.io.File;
import java.util.Locale;

public class MainActivity extends AppCompatActivity {
    private static final String TAG = "EdgeShield_ET";
    private static final String SVM_TAG = "EdgeShieldSVM";
    private static final String MODEL_NAME = "llama_3_2_1b_xnnpack_spinquant.pte";
    private static final String TOKENIZER_NAME = "tokenizer.json";
    private static final long MODEL_BYTES = 1082483584L;
    private static final long TOKENIZER_BYTES = 17209920L;

    private volatile EdgeShieldSvm svmModel;
    private volatile boolean svmReady = false;
    private volatile String svmError = null;

    private EditText smsInput;
    private TextView statusText;
    private TextView riskText;
    private TextView outputText;
    private TextView timingText;
    private Button loadButton;
    private Button generateButton;
    private Button pasteButton;
    private Button clearButton;
    private TextView characterCount;

    private volatile LlmModule llama;
    private volatile boolean busy = false;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        // UI resources are separate from the verified SVM and ExecuTorch engine.
        setContentView(R.layout.activity_main);
        getWindow().setStatusBarColor(Color.rgb(248, 250, 255));
        getWindow().setNavigationBarColor(Color.rgb(248, 250, 255));
        getWindow().getDecorView().setSystemUiVisibility(
                View.SYSTEM_UI_FLAG_LIGHT_STATUS_BAR | View.SYSTEM_UI_FLAG_LIGHT_NAVIGATION_BAR);

        smsInput = findViewById(R.id.smsInput);
        statusText = findViewById(R.id.statusText);
        riskText = findViewById(R.id.riskText);
        outputText = findViewById(R.id.outputText);
        timingText = findViewById(R.id.timingText);
        loadButton = findViewById(R.id.loadButton);
        generateButton = findViewById(R.id.generateButton);
        pasteButton = findViewById(R.id.pasteButton);
        clearButton = findViewById(R.id.clearButton);
        characterCount = findViewById(R.id.characterCount);

        statusText.setText("Preparing offline classifier…");
        riskText.setText("Awaiting your first analysis");
        outputText.setText("Your on-device explanation will appear here.");
        generateButton.setEnabled(false);

        // A test SMS is deliberately not prefilled in the production-facing UI.
        smsInput.addTextChangedListener(new TextWatcher() {
            @Override public void beforeTextChanged(CharSequence s, int start, int count, int after) { }
            @Override public void onTextChanged(CharSequence s, int start, int before, int count) {
                characterCount.setText(s.length() + " characters");
            }
            @Override public void afterTextChanged(Editable s) { }
        });

        pasteButton.setOnClickListener(v -> {
            ClipboardManager clipboard = (ClipboardManager) getSystemService(Context.CLIPBOARD_SERVICE);
            if (clipboard == null || !clipboard.hasPrimaryClip()) {
                Toast.makeText(this, "Clipboard is empty", Toast.LENGTH_SHORT).show();
                return;
            }
            ClipData data = clipboard.getPrimaryClip();
            if (data != null && data.getItemCount() > 0) {
                CharSequence text = data.getItemAt(0).coerceToText(this);
                if (text != null) smsInput.setText(text);
            }
        });
        clearButton.setOnClickListener(v -> smsInput.setText(""));
        loadButton.setOnClickListener(v -> loadModel());
        generateButton.setOnClickListener(v -> generateExplanation());

        initializeSvm();
    }

    private void updateGenerateButton() {
        generateButton.setEnabled(!busy && llama != null && svmReady);
    }

    private void initializeSvm() {
        Log.i(SVM_TAG, "Starting SVM initialization");
        new Thread(() -> {
            try {
                EdgeShieldSvm loaded = new EdgeShieldSvm(getApplicationContext());
                svmModel = loaded;
                svmReady = true;
                svmError = null;
                Log.i(SVM_TAG, "SVM model loaded successfully");
                runOnUiThread(() -> {
                    riskText.setText("SVM classifier ready");
                    statusText.setText(llama == null ? "SVM ready | Load Llama model" : "Both models ready");
                    updateGenerateButton();
                });
            } catch (Exception | LinkageError e) {
                svmReady = false;
                svmError = e.toString();
                Log.e(SVM_TAG, "SVM initialization failed", e);
                runOnUiThread(() -> {
                    riskText.setText("SVM failed to load: " + svmError);
                    statusText.setText("SVM initialization failed");
                    updateGenerateButton();
                });
            }
        }, "SVM-Loader").start();
    }

    private EdgeShieldSvm.Result classifySms(String sms) {
        EdgeShieldSvm model = svmModel;
        if (!svmReady || model == null) {
            throw new IllegalStateException("SVM model is not ready");
        }
        return model.predict(sms);
    }

    private void loadModel() {
        if (busy || llama != null) return;
        busy = true;
        loadButton.setEnabled(false);
        updateGenerateButton();
        statusText.setText("Loading Llama model...");

        new Thread(() -> {
            try {
                File model = new File(getFilesDir(), MODEL_NAME);
                File tokenizer = new File(getFilesDir(), TOKENIZER_NAME);
                if (!model.exists() || !tokenizer.exists()) {
                    throw new Exception("Model or tokenizer missing from app storage");
                }
                if (model.length() != MODEL_BYTES || tokenizer.length() != TOKENIZER_BYTES) {
                    throw new Exception("Model file size mismatch");
                }

                long start = System.nanoTime();
                LlmModule loaded = new LlmModule(
                        model.getAbsolutePath(), tokenizer.getAbsolutePath(), 0.0f);
                loaded.load();
                llama = loaded;
                double seconds = (System.nanoTime() - start) / 1e9;
                Log.i(TAG, "Model loaded in " + seconds + " seconds");

                runOnUiThread(() -> {
                    statusText.setText(String.format(Locale.US,
                            "Llama loaded in %.2f seconds | SVM %s",
                            seconds, svmReady ? "ready" : "not ready"));
                    loadButton.setText("Model Loaded");
                    busy = false;
                    updateGenerateButton();
                });
            } catch (Exception | LinkageError e) {
                Log.e(TAG, "Model loading error", e);
                showError("Model loading failed", e);
            }
        }, "Llama-Loader").start();
    }

    private void generateExplanation() {
        if (busy || llama == null || !svmReady) {
            statusText.setText("Wait until both models are ready.");
            return;
        }

        final String sms = smsInput.getText().toString().trim();
        if (sms.isEmpty()) {
            statusText.setText("Please enter an SMS.");
            return;
        }

        busy = true;
        updateGenerateButton();
        loadButton.setEnabled(false);
        outputText.setText("");
        timingText.setText("");
        riskText.setText("Classifying SMS with SVM...");
        statusText.setText("Classifying SMS...");

        new Thread(() -> {
            try {
                EdgeShieldSvm.Result result = classifySms(sms);
                final boolean scam = result.prediction == 1;
                final String classification = scam ? "SCAM" : "LEGITIMATE";
                final double margin = result.score;

                Log.i(SVM_TAG, "Classification=" + classification + ", margin=" + margin);
                runOnUiThread(() -> {
                    riskText.setText(String.format(Locale.US,
                            "%s  ·  SVM classification\nDecision margin: %+.5f (not a probability)",
                            classification, margin));
                    riskText.setTextColor(scam ? Color.rgb(171, 47, 65) : Color.rgb(24, 115, 100));
                    statusText.setText("Generating Llama explanation...");
                });

                // The SVM supplies a prediction, while Llama generates every
                // explanation from the actual SMS. No canned explanations.
                // Avoid visible delimiters such as <<<SMS that the 1B model
                // may mistakenly describe as part of the message.
                // Keep the instructions short: small local LLMs can copy
                // complex bullet-format instructions instead of answering.
                String prompt =
                        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                                + "You help ordinary people understand SMS messages. "
                                + "Explain this exact SMS in simple, natural English. "
                                + "Use three short bullet points: what the message says, "
                                + "why it looks safe or suspicious, and what the person should do. "
                                + "Describe only details actually in the SMS. "
                                + "Do not invent risks or call everyday conversation phishing. "
                                + "A scam prediction is not proof. Never recommend sharing an OTP. "
                                + "Write useful words, not just dashes or headings. "
                                + "Keep it under 100 words."
                                + "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
                                + "SMS: " + sms + "\n"
                                + "Classifier result: " + classification + ".\n"
                                + "Explain this message for a non-technical phone user."
                                + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n";

                final StringBuilder generated = new StringBuilder();
                final long start = System.nanoTime();
                final boolean[] runtimeError = {false};

                LlmGenerationConfig config = LlmGenerationConfig.create()
                        .seqLen(768)
                        .maxNewTokens(145)
                        .echo(false)
                        .build();

                LlmCallback callback = new LlmCallback() {
                    @Override
                    public void onResult(String token) {
                        String cleaned = token
                                .replace("<|eot_id|>", "")
                                .replace("<|end_of_text|>", "")
                                .replace("<|start_header_id|>", "")
                                .replace("<|end_header_id|>", "")
                                .replace("<|eom_id|>", "");
                        if (cleaned.isEmpty()) return;
                        generated.append(cleaned);
                        // Stream genuine model output. Do not show punctuation-only
                        // fragments such as "---" as a completed explanation.
                        final String snapshot = generated.toString();
                        if (hasMeaningfulText(snapshot)) {
                            runOnUiThread(() -> outputText.setText(snapshot));
                        }
                    }

                    @Override
                    public void onStats(String statsJson) {
                        Log.i(TAG, "Generation stats: " + statsJson);
                    }

                    @Override
                    public void onError(int errorCode, String message) {
                        runtimeError[0] = true;
                        Log.e(TAG, "Generation error " + errorCode + ": " + message);
                        runOnUiThread(() -> statusText.setText(
                                "Generation error " + errorCode + ": " + message));
                    }
                };

                llama.generate(prompt, config, callback);
                double seconds = (System.nanoTime() - start) / 1e9;
                Log.i(TAG, "Generation elapsed: " + seconds);
                Log.i(TAG, "Generated text: " + generated);
                final String rawExplanation = generated.toString().trim();
                // Preserve the exact model-generated text; never substitute templates.
                // An empty/error response is reported as an error, not disguised
                // as an AI-generated explanation.
                runOnUiThread(() -> {
                    if (!runtimeError[0]) {
                        if (!hasMeaningfulText(rawExplanation)) {
                            statusText.setText("Llama did not produce a usable explanation");
                            outputText.setText("The language model returned no meaningful explanation. Please retry.");
                            Log.w(TAG, "Unusable Llama output: " + rawExplanation);
                        } else {
                            outputText.setText(rawExplanation);
                            statusText.setText("Analysis completed · on-device AI");
                        }
                        timingText.setText(String.format(Locale.US,
                                "Generation time: %.2f seconds", seconds));
                    }
                    busy = false;
                    updateGenerateButton();
                });
            } catch (Exception | LinkageError e) {
                Log.e(TAG, "Classification or generation failed", e);
                showError("Analysis failed", e);
            }
        }, "SVM-Llama-Analysis").start();
    }

    // Checks that an explanation contains actual words, not just bullet marks.
    // This does not replace Llama's answer with a hardcoded explanation.
    private static boolean hasMeaningfulText(String text) {
        int letters = 0;
        for (int i = 0; i < text.length(); i++) {
            if (Character.isLetter(text.charAt(i))) letters++;
            if (letters >= 12) return true;
        }
        return false;
    }

    private void showError(String heading, Throwable error) {
        String message = error.getMessage();
        if (message == null) message = error.toString();
        final String finalMessage = message;
        runOnUiThread(() -> {
            statusText.setText(heading);
            outputText.setText(finalMessage);
            busy = false;
            loadButton.setEnabled(llama == null);
            updateGenerateButton();
        });
    }
}
