import os
import subprocess

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs"))
SUMMARY = []

def convert_model(model_name, saved_model_dir):
    tfjs_output_dir = os.path.join(os.path.dirname(saved_model_dir), "tfjs_model")
    os.makedirs(tfjs_output_dir, exist_ok=True)
    cmd = [
        "tensorflowjs_converter",
        "--input_format=tf_saved_model",
        saved_model_dir,
        tfjs_output_dir,
    ]
    print(f"🔄 Converting [{model_name}] to tfjs...")
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"✅ [{model_name}] converted to {tfjs_output_dir}")
        SUMMARY.append((model_name, "Converted", tfjs_output_dir))
    except subprocess.CalledProcessError as e:
        print(f"❌ ERROR converting [{model_name}]:\n{e.stderr.decode()}")
        SUMMARY.append((model_name, "Failed", str(e)))

def main():
    print(f"🔍 Scanning '{ROOT}' for models...")
    found_any = False
    for model_name in os.listdir(ROOT):
        model_dir = os.path.join(ROOT, model_name)
        saved_model_dir = os.path.join(model_dir, "saved_model")
        if os.path.isdir(saved_model_dir):
            found_any = True
            convert_model(model_name, saved_model_dir)
        else:
            SUMMARY.append((model_name, "Skipped (no saved_model)", ""))
    if not found_any:
        print("⚠️  No models with 'saved_model' found in output/. Nothing to convert.")
    print("\n====== Conversion Summary ======")
    for model, status, path in SUMMARY:
        print(f"{model.ljust(20)} | {status.ljust(18)} | {path}")
    print("================================\n")

if __name__ == "__main__":
    main()
