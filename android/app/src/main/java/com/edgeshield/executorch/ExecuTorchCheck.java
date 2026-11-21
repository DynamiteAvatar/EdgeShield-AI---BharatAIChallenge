package com.edgeshield.executorch;
import org.pytorch.executorch.extension.llm.LlmModule;
import org.pytorch.executorch.extension.llm.LlmGenerationConfig;
import org.pytorch.executorch.extension.llm.LlmCallback;

public class ExecuTorchCheck {

    private LlmModule module;

    public boolean isRuntimeApiAvailable() {
        return LlmModule.class != null
                && LlmGenerationConfig.class != null
                && LlmCallback.class != null;
    }
}
