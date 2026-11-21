
package com.edgeshield.executorch;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.regex.Pattern;

public final class ScamIntelligence {

    private ScamIntelligence() {}

    public static final class Result {
        public final String riskLevel;
        public final int score;
        public final List<String> evidence;

        Result(String riskLevel, int score, List<String> evidence) {
            this.riskLevel = riskLevel;
            this.score = score;
            this.evidence = evidence;
        }

        public String toPromptEvidence() {
            StringBuilder sb = new StringBuilder();

            sb.append("Rule-based risk assessment: ")
                    .append(riskLevel)
                    .append("\n");

            sb.append("Matched indicators:\n");

            if (evidence.isEmpty()) {
                sb.append("- No predefined scam indicators matched.\n");
            } else {
                for (String item : evidence) {
                    sb.append("- ").append(item).append("\n");
                }
            }

            sb.append(
                    "These are heuristic indicators, not proof of fraud.\n"
            );

            return sb.toString();
        }
    }

    private static boolean matches(
            String text, String regex) {

        return Pattern.compile(
                regex,
                Pattern.CASE_INSENSITIVE
                        | Pattern.UNICODE_CASE
        ).matcher(text).find();
    }

    public static Result analyze(String sms) {

        String text = sms.toLowerCase(Locale.ROOT);

        int score = 0;
        List<String> evidence = new ArrayList<>();

        boolean otp = matches(
                text,
                "\\b(otp|one.time.password|verification code)\\b"
        );

        boolean urgency = matches(
                text,
                "\\b(immediately|urgent|within 24 hours|"
                        + "act now|last chance)\\b"
        );

        boolean accountThreat = matches(
                text,
                "\\b(blocked|suspended|deactivated|"
                        + "frozen|locked)\\b"
        );

        boolean financialContext = matches(
                text,
                "\\b(bank|account|upi|payment|wallet|"
                        + "kyc|card)\\b"
        );

        boolean credentialRequest = matches(
                text,
                "\\b(share|send|provide|reply with|"
                        + "verify|submit|enter)\\b"
        ) && otp;

        boolean link = matches(
                text,
                "(https?://|www\\.|\\b[a-z0-9-]+"
                        + "\\.(com|in|net|org|xyz|top|click)"
                        + "(?:/|\\b))"
        );

        if (otp) {
            score += 15;
            evidence.add("Message mentions an OTP or verification code");
        }

        if (urgency) {
            score += 20;
            evidence.add("Message uses urgent language");
        }

        if (accountThreat) {
            score += 20;
            evidence.add("Message threatens account restriction");
        }

        if (financialContext) {
            score += 10;
            evidence.add("Message refers to financial services");
        }

        if (credentialRequest) {
            score += 30;
            evidence.add(
                    "Message appears to request OTP verification"
            );
        }

        if (link) {
            score += 10;
            evidence.add("Message contains a web link");
        }

        if (otp && urgency && accountThreat) {
            score += 20;
            evidence.add(
                    "OTP, urgency, and account threat appear together"
            );
        }

        score = Math.min(score, 100);

        String riskLevel;

        if (score >= 60) {
            riskLevel = "HIGH";
        } else if (score >= 30) {
            riskLevel = "MEDIUM";
        } else {
            riskLevel = "LOW";
        }

        return new Result(riskLevel, score, evidence);
    }
}
