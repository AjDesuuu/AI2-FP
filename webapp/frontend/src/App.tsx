import { useState } from "react";
import "./App.css";

const LINK = "/api/classify";

function App() {
  const [file, setFile] = useState<File | null>(null);
  const [output, setOutput] = useState<Blob | null>(null);

  async function handleSubmit(event: React.MouseEvent<HTMLButtonElement, MouseEvent>) {
    event.preventDefault();
    if (!file) return;

    const formData = new FormData();
    formData.set("file", file);

    try {
      const response = await fetch(LINK, {
        method: "POST",
        body: formData,
        headers: { "Access-Control-Request-Method": "POST" },
      });

      if (response.headers.get("content-type") !== "image/png") throw new Error("Invalid file type");

      setOutput(await response.blob());
    } catch (err) {
      console.error(err);
    }
  }

  return (
    <>
      <h1>YOLOv8 Road Obstruction Detection</h1>
      <p>Select your file here: </p>
      <input
        type="file"
        accept="image/jpeg, image/png"
        onChange={(event) => {
          const file = event.target.files?.[0];
          if (file) setFile(file);
        }}
      />

      {file !== null && (
        <>
          <img className="demo_image" src={URL.createObjectURL(file)} alt="File Preview" />
          <button onClick={handleSubmit}>Submit</button>
        </>
      )}

      {output !== null && (
        <>
          <img className="demo_image" src={URL.createObjectURL(output)} alt="Output Preview" />
        </>
      )}
    </>
  );
}

export default App;
