
package com.edgeshield.executorch;

import android.content.Context;
import android.util.Log;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;

public class SvmParityTest {

    private static final String TAG = "EdgeShieldParity";

    public static void run(Context context) {

        try {
            EdgeShieldSvm svm =
                    new EdgeShieldSvm(context);

            JSONArray references;

            try (InputStream input =
                         context.getAssets().open(
                                 "parity_reference.json"
                         )) {

                ByteArrayOutputStream output =
                        new ByteArrayOutputStream();

                byte[] buffer = new byte[8192];
                int n;

                while ((n = input.read(buffer)) != -1) {
                    output.write(buffer, 0, n);
                }

                references = new JSONArray(
                        new String(
                                output.toByteArray(),
                                StandardCharsets.UTF_8
                        )
                );
            }

            int passed = 0;
            int failed = 0;

            double maxDifference = 0.0;

            // Numerical tolerance for comparing
            // Java and Python decision scores.
            final double SCORE_TOLERANCE = 1e-4;

            for (int i = 0;
                 i < references.length();
                 i++) {

                JSONObject item =
                        references.getJSONObject(i);

                String sms =
                        item.getString("text");

                int expectedPrediction =
                        item.getInt("python_prediction");

                double expectedScore =
                        item.getDouble(
                                "python_decision_score"
                        );

                EdgeShieldSvm.Result actual =
                        svm.predict(sms);

                double difference = Math.abs(
                        actual.score - expectedScore
                );

                maxDifference = Math.max(
                        maxDifference,
                        difference
                );

                boolean predictionMatches =
                        actual.prediction ==
                                expectedPrediction;

                boolean scoreMatches =
                        difference <= SCORE_TOLERANCE;

                if (predictionMatches && scoreMatches) {
                    passed++;
                } else {
                    failed++;

                    Log.e(
                            TAG,
                            "FAIL id=" + item.getInt("id")
                                    + " language="
                                    + item.getString("language")
                                    + " expectedPrediction="
                                    + expectedPrediction
                                    + " actualPrediction="
                                    + actual.prediction
                                    + " expectedScore="
                                    + expectedScore
                                    + " actualScore="
                                    + actual.score
                                    + " difference="
                                    + difference
                    );
                }
            }

            Log.i(TAG, "================================");
            Log.i(TAG, "EDGESHIELD SVM PARITY RESULTS");
            Log.i(TAG, "================================");

            Log.i(
                    TAG,
                    "Total references: "
                            + references.length()
            );

            Log.i(TAG, "Passed: " + passed);
            Log.i(TAG, "Failed: " + failed);

            Log.i(
                    TAG,
                    "Maximum score difference: "
                            + maxDifference
            );

            if (failed == 0) {
                Log.i(
                        TAG,
                        "PARITY TEST PASSED"
                );
            } else {
                Log.e(
                        TAG,
                        "PARITY TEST FAILED"
                );
            }

        } catch (Exception e) {
            Log.e(
                    TAG,
                    "Parity test error",
                    e
            );
        }
    }
}
