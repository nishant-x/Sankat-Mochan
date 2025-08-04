import React, { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import axios from "axios";
import { DotLottieReact } from "@lottiefiles/dotlottie-react";
import { ToastContainer, toast } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";
import "../AllLogin.css";

const Login = () => {
  const [formData, setFormData] = useState({ email: "", password: "" });
  const [error, setError] = useState("");
  const navigate = useNavigate();

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!formData.email || !formData.password) {
      setError("Both fields are required!");
      toast.error("Both fields are required!", {
        position: "top-right",
        autoClose: 3000,
        hideProgressBar: false,
        closeOnClick: true,
        pauseOnHover: true,
        draggable: true,
      });
      return;
    }

    try {
      const response = await axios.post(
        "http://localhost:5000/api/login/inventory_manager_login",
        formData
      );

      const userDetails = response.data;
      localStorage.setItem("userDetails", JSON.stringify(userDetails));

      toast.success("Login Successful!", {
        position: "top-right",
        autoClose: 3000,
        hideProgressBar: false,
        closeOnClick: true,
        pauseOnHover: true,
        draggable: true,
      });

      navigate("/pharmacy-dashboard", { state: { userDetails } });
    } catch (error) {
      setError("Invalid email or password");
      toast.error("Invalid email or password", {
        position: "top-right",
        autoClose: 3000,
        hideProgressBar: false,
        closeOnClick: true,
        pauseOnHover: true,
        draggable: true,
      });
      console.error("Login error:", error);
    }
  };

  return (
    <div className="all-login-container">
      {/* Toast Container */}
      <ToastContainer />

      <h1>Pharmacy Login</h1>
      <form onSubmit={handleSubmit}>
        <DotLottieReact
          src="https://lottie.host/e088efab-c64c-4256-9cbd-0cc53bc83a2a/ewP7msP3mq.lottie"
          loop
          autoplay
        />
        <div className="all-login-input-group">
          <label>Email:</label>
          <input
            type="email"
            name="email"
            placeholder="Enter email"
            value={formData.email}
            onChange={handleChange}
            required
          />
        </div>

        <div className="all-login-input-group">
          <label>Password:</label>
          <input
            type="password"
            name="password"
            placeholder="Enter password"
            value={formData.password}
            onChange={handleChange}
            required
          />
        </div>

        {error && <p className="all-login-error">{error}</p>}

        <button type="submit" className="all-login-button">
          Login
        </button>
        <p className="all-login-register-link">
          Don't have an account? <Link to="/pharmacy-registration">Register Now</Link>
        </p>
      </form>
    </div>
  );
};

export default Login;