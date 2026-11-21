
package com.edgeshield.executorch;

import android.content.Context;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.zip.GZIPInputStream;

public class EdgeShieldSvm {

    private static final Pattern WORD_PATTERN =
            Pattern.compile(
                    "[\\p{L}\\p{N}_]{2,}"
            );

    private static final Pattern WHITESPACE =
            Pattern.compile("\\s\\s+");

    private final Vectorizer word;
    private final Vectorizer character;
    private final double intercept;

    public static class Result {
        public final int prediction;
        public final double score;

        Result(int prediction, double score) {
            this.prediction = prediction;
            this.score = score;
        }

        public String label() {
            return prediction == 1 ? "SCAM" : "HAM";
        }
    }

    private static class Feature {
        final double idf;
        final double weight;

        Feature(double idf, double weight) {
            this.idf = idf;
            this.weight = weight;
        }
    }

    private static class Vectorizer {

        final String analyzer;
        final int minN;
        final int maxN;
        final boolean sublinearTf;

        final HashMap<String, Feature> features =
                new HashMap<>();

        Vectorizer(JSONObject json) throws Exception {

            analyzer = json.getString("analyzer");

            JSONArray range =
                    json.getJSONArray("ngram_range");

            minN = range.getInt(0);
            maxN = range.getInt(1);

            sublinearTf =
                    json.getBoolean("sublinear_tf");

            JSONArray vocabulary =
                    json.getJSONArray("vocabulary");

            JSONArray idf =
                    json.getJSONArray("idf");

            JSONArray weights =
                    json.getJSONArray("weights");

            if (vocabulary.length() != idf.length()
                    || idf.length() != weights.length()) {
                throw new IllegalArgumentException(
                        "Feature dimensions mismatch"
                );
            }

            for (int i = 0; i < vocabulary.length(); i++) {
                features.put(
                        vocabulary.getString(i),
                        new Feature(
                                idf.getDouble(i),
                                weights.getDouble(i)
                        )
                );
            }
        }

        private ArrayList<String> tokenizeWords(
                String text
        ) {
            ArrayList<String> tokens =
                    new ArrayList<>();

            Matcher matcher =
                    WORD_PATTERN.matcher(text);

            while (matcher.find()) {
                tokens.add(matcher.group());
            }

            return tokens;
        }

        private void addCount(
                HashMap<String, Integer> counts,
                String feature
        ) {
            if (!features.containsKey(feature)) {
                return;
            }

            Integer current = counts.get(feature);

            counts.put(
                    feature,
                    current == null ? 1 : current + 1
            );
        }

        private HashMap<String, Integer> extract(
                String original
        ) {
            HashMap<String, Integer> counts =
                    new HashMap<>();

            // Python's str.lower() and Java lowercase
            // are not perfectly identical for all Unicode.
            String text =
                    original.toLowerCase(Locale.ROOT);

            if (analyzer.equals("word")) {

                ArrayList<String> tokens =
                        tokenizeWords(text);

                for (int n = minN; n <= maxN; n++) {
                    for (int i = 0;
                         i + n <= tokens.size();
                         i++) {

                        StringBuilder gram =
                                new StringBuilder();

                        for (int j = 0; j < n; j++) {
                            if (j > 0) {
                                gram.append(' ');
                            }

                            gram.append(tokens.get(i + j));
                        }

                        addCount(
                                counts,
                                gram.toString()
                        );
                    }
                }

            } else if (analyzer.equals("char")) {

                // Match scikit-learn's whitespace
                // collapsing for character n-grams.
                text = WHITESPACE.matcher(text)
                        .replaceAll(" ");

                int[] codepoints =
                        text.codePoints().toArray();

                for (int n = minN; n <= maxN; n++) {
                    for (int i = 0;
                         i + n <= codepoints.length;
                         i++) {

                        String gram = new String(
                                codepoints,
                                i,
                                n
                        );

                        addCount(counts, gram);
                    }
                }

            } else {
                throw new IllegalStateException(
                        "Unsupported analyzer: " + analyzer
                );
            }

            return counts;
        }

        double score(String text) {

            HashMap<String, Integer> counts =
                    extract(text);

            double squaredNorm = 0.0;
            double weightedSum = 0.0;

            for (Map.Entry<String, Integer> entry
                    : counts.entrySet()) {

                Feature feature =
                        features.get(entry.getKey());

                int frequency = entry.getValue();

                double tf = sublinearTf
                        ? 1.0 + Math.log(frequency)
                        : frequency;

                double value = tf * feature.idf;

                squaredNorm += value * value;

                weightedSum += value * feature.weight;
            }

            if (squaredNorm == 0.0) {
                return 0.0;
            }

            return weightedSum / Math.sqrt(squaredNorm);
        }
    }

    public EdgeShieldSvm(Context context)
            throws Exception {

        byte[] bytes;

        try (
                InputStream raw = context.getAssets().open(
                        "edgeshield_svm.json"
                )
        ) {
            bytes = readAll(raw);
        }

        JSONObject model = new JSONObject(
                new String(
                        bytes,
                        StandardCharsets.UTF_8
                )
        );

        if (!model.getString("format").equals(
                "edgeshield-linear-svm-v1"
        )) {
            throw new IllegalArgumentException(
                    "Unexpected model format"
            );
        }

        JSONArray classes =
                model.getJSONArray("classes");

        if (classes.getInt(0) != 0
                || classes.getInt(1) != 1) {
            throw new IllegalArgumentException(
                    "Unexpected class ordering"
            );
        }

        intercept = model.getDouble("intercept");

        JSONObject union =
                model.getJSONObject("feature_union");

        if (!union.isNull("transformer_weights")) {
            throw new IllegalArgumentException(
                    "Weighted FeatureUnion unsupported"
            );
        }

        word = new Vectorizer(
                union.getJSONObject("word")
        );

        character = new Vectorizer(
                union.getJSONObject("char")
        );
    }

    private static byte[] readAll(InputStream input)
            throws IOException {

        ByteArrayOutputStream output =
                new ByteArrayOutputStream();

        byte[] buffer = new byte[8192];

        int count;

        while ((count = input.read(buffer)) != -1) {
            output.write(buffer, 0, count);
        }

        return output.toByteArray();
    }

    public Result predict(String sms) {

        if (sms == null) {
            throw new IllegalArgumentException(
                    "SMS cannot be null"
            );
        }

        double score =
                intercept
                        + word.score(sms)
                        + character.score(sms);

        return new Result(
                score > 0.0 ? 1 : 0,
                score
        );
    }
}
