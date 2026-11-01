import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import Admin from "./Admin";
import "./styles.css";

// ?admin in the URL opens the (password-gated) admin view.
const isAdmin = new URLSearchParams(window.location.search).has("admin");

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    {isAdmin ? <Admin /> : <App />}
  </React.StrictMode>,
);
