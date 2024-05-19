import React, { useState } from "react";
import axios from "axios";

import {
  BrowserRouter as Router,
  Route,
  Routes,
  Outlet,
} from "react-router-dom";

import "bootstrap/dist/css/bootstrap.min.css";
import NavbarComponent from "./components/NavbarComponent";
import Lowpass from "./components/Lowpass";
import Butterworth from "./components/Butterworth";
import Histogram from "./components/Histogram";
import Highpass from "./components/Highpass";
import Feature from "./components/Feature";

function App() {
  return (
    <Router>
      <NavbarComponent />
      <Routes>
        <Route path="/" element={<Lowpass />} />
        <Route path="/butterworth" element={<Butterworth />} />
        <Route path="/histogram" element={<Histogram />} />

        <Route path="/highpass" element={<Highpass />} />
        <Route path="/feature" element={<Feature />} />
      </Routes>
    </Router>
  );
}

export default App;
