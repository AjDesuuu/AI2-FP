import { useState } from "react";

const LINK = "/api/classify";
const models = {
  model_15_2: {
    label: "Model 15v2 (best model)",
    description:
      "yolov8m weight, 47 epochs, batch 128, 100k images, Adam optimizer with uses Cosine Anneling (CA), dropout=0.3",
  },
  model_1: {
    label: "Model 1",
    description: "yolov8n weight, 5 epochs, batch 32",
  },
  model_2: {
    label: "Model 2",
    description: " yolov8n weight, 5 epochs, batch 16",
  },
  model_3: {
    label: "Model 3",
    description: "yolov8n weight, 10 epochs, batch 32",
  },
  model_4: {
    label: "Model 4",
    description: "yolov8s weight, 10 epochs, batch 32",
  },
  model_5: {
    label: "Model 5",
    description: "yolov8s weight, 100 epochs, batch 32",
  },
  model_6: {
    label: "Model 6",
    description: "yolov8n weight, 5 epochs, batch 32, 100k images, cpu trained",
  },
  model_7: {
    label: "Model 7",
    description: "yolov8n weight, 5 epochs, batch 32, 100k images, gpu trained",
  },
  model_9: {
    label: "Model 9",
    description: "yolov8n weight, 5 epochs, batch 32, 10k images",
  },
  model_8: {
    label: "Model 8",
    description: "yolov8n weight, 10 epochs, batch 32, 1500 imagess (randomly chosen)",
  },
  model_10: {
    label: "Model 10",
    description: "yolov8n weight, 10 epochs, batch 32, 1500 imagess (handpicked)",
  },
  model_10_5: {
    label: "Model 10_5",
    description: "yolov8n weight, 10 epochs, batch 32, 1500 images (augmented with SMOTE to fix class imbalance)",
  },
  model_11: {
    label: "Model 11",
    description: "yolov8n weight, 50 epochs, batch 32, Adam optimizer, cls 1.5",
  },
  model_12: {
    label: "Model 12",
    description: "yolov8n weight, 25 epochs, batch 32, Adam optimizer, cls 1.5, uses CA",
  },
  model_13: {
    label: "Model 13",
    description: "yolov8s weight, 15 epochs, batch 32, dropout=0.3",
  },
  model_14: {
    label: "Model 14",
    description: "yolov8m weight, 15 epochs, batch 64, 100k images, Adam optimizer with CA",
  },
  model_14_5: {
    label: "Model 14_5",
    description: "yolov8m weight, 15 epochs, batch 32, 1.5k images (augmented), Adam optimizer with CA",
  },
  model_15: {
    label: "Model 15",
    description:
      "yolov8m weight, 5 epochs, batch 128, 1.5k images (augmented), Adam optimizer with uses CA, dropout=0.3",
  },
};

function App() {
  const [selectedModel, setSelectedModel] = useState<string>("model_15_2");
  const [file, setFile] = useState<File | null>(null);
  const [output, setOutput] = useState<Blob | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  async function handleSubmit(event: React.MouseEvent<HTMLButtonElement, MouseEvent>) {
    event.preventDefault();
    if (!file) return;

    const formData = new FormData();
    formData.set("file", file);
    formData.set("model", selectedModel);
    setIsSubmitting(true);

    try {
      const response = await fetch(LINK, {
        method: "POST",
        body: formData,
        headers: { "Access-Control-Request-Method": "POST" },
      });

      if (response.headers.get("content-type") !== "image/png") throw new Error("Invalid file type");

      setOutput(await response.blob());
    } catch (err) {
      alert("an error has occured, check the console for more details");
      console.error(err);
    }

    setIsSubmitting(false);
  }

  return (
    <div style={{ display: "flex", gap: 20 }} className="main_page">
      <div style={{ flex: 1, padding: 20 }}>
        <h1 style={{ fontSize: 32, marginBottom: 20 }}>YOLOv8 Road Object Analysis</h1>
        <p style={{ marginBottom: 30 }}>
          This website is a live demonstration of various YOLOv8 models aimed at detecting objects. Canva presentation
          can be found{" "}
          <a
            target="_blank"
            rel="noreferrer"
            href="https://www.canva.com/design/DAGXiwrcX5E/JC25ZTN9MRc8NrK4HMKqoA/edit?utm_content=DAGXiwrcX5E&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton"
          >
            here
          </a>{" "}
          and Google Colab can be found{" "}
          <a
            target="_blank"
            rel="noreferrer"
            href="https://colab.research.google.com/drive/1jXiOR3KoMtEoPWEbU1718O0ocoA31yAx?usp=sharing"
          >
            here
          </a>{" "}
          (readonly).
        </p>

        <h2 style={{ marginBottom: 10 }}>Select which model to use:</h2>
        <div style={{ display: "flex", gap: 10, flexDirection: "column" }}>
          {Object.entries(models).map(([key, value]) => (
            <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
              <input type="radio" id={key} checked={key == selectedModel} onChange={() => setSelectedModel(key)} />

              <p style={{ margin: 0, padding: 0 }}>
                <b>{value.label}</b> - {value.description}
              </p>
            </div>
          ))}
        </div>
      </div>

      <div style={{ flex: 1, padding: 20 }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 20 }}>
          <p style={{ fontWeight: "bold" }}>Select your file here: </p>
          <input
            type="file"
            accept="image/jpeg, image/png"
            onChange={(event) => {
              const file = event.target.files?.[0];
              if (file) setFile(file);
            }}
          />
        </div>

        {file !== null && (
          <div style={{ marginBottom: 30 }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
              <h2>Your image</h2>
              <button onClick={handleSubmit} disabled={isSubmitting}>
                {isSubmitting ? "Submitting..." : "Submit"}
              </button>
            </div>

            <img className="demo_image" src={URL.createObjectURL(file)} alt="File Preview" />
          </div>
        )}

        {output !== null && (
          <div>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
              <h2>Output</h2>
            </div>

            <img className="demo_image" src={URL.createObjectURL(output)} alt="Output Preview" />
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
