import axios from "axios";
import React, { useState } from "react";

const Lowpass = () => {
  const [pythonResult, setPythonResult] = useState({});
  const [selectedFile, setSelectedFile] = useState("");

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    const path = `d:/images/${file.name}`;
    setSelectedFile(path);
  };

  const uploadFile = async (e) => {
    e.preventDefault();
    try {
      const response = await axios.post(
        "http://127.0.0.1:5000/api/lowpass",
        { image: selectedFile },
        {
          headers: {
            "Content-Type": "application/json",
          },
        }
      );

      console.log(response.data);
      setPythonResult({
        image: response.data._image,
        processed_image: response.data.processed_image,
      });
    } catch (error) {
      console.error("Error:", error);
    }
  };

  return (
    <form onSubmit={uploadFile} style={{ padding: "20px" }}>
      <h1 style={{ padding: "2rem" }}>Lowpass Filter</h1>
      <input type="file" accept="image/*" onChange={handleFileChange} />
      <button type="submit">Upload File</button>

      {pythonResult.image && pythonResult.processed_image && (
        <div style={{ display: "flex", gap: "10px", padding: "10px" }}>
          <img
            style={{ width: "500px", height: "500px" }}
            src={require(`../../../uploads/${pythonResult.image}`)}
            alt="__image"
          />
          <img
            style={{ width: "500px", height: "500px" }}
            src={require(`../../../uploads/${pythonResult.processed_image}`)}
            alt="__image"
          />{" "}
        </div>
      )}
    </form>
  );
};

export default Lowpass;
