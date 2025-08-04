import React, { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { DotLottieReact } from "@lottiefiles/dotlottie-react";
import axios from "axios";
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
        "http://localhost:5000/api/login/admin_login",
        formData
      );

      // Assuming 'response.data' contains the user details
      const userDetails = response.data;
      localStorage.setItem("userDetails", JSON.stringify(userDetails));

      // Show success toast
      toast.success("Login Successful!", {
        position: "top-right",
        autoClose: 3000,
        hideProgressBar: false,
        closeOnClick: true,
        pauseOnHover: true,
        draggable: true,
      });

      // Navigate based on user role
      if (userDetails.userRole === "Receptionist") {
        navigate("/receptionist-admin-dashboard", { state: { userDetails } });
      } else if (userDetails.userRole === "HR Manager") {
        navigate("/hrmanager-admin-dashboard", { state: { userDetails } });
      } else if (userDetails.userRole === "System Administrator") {
        navigate("/systemadministrator-admin-dashboard", { state: { userDetails } });
      }
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

      {/* <div className="svg"></div> <DotLottieReact
      src="https://lottie.host/aefd25a9-0761-4192-8a02-a1c591fc71a2/kAggrIqva7.lottie"
      loop
      autoplay/> */}
      <h1>Admin Login</h1>

      <form onSubmit={handleSubmit}>
        <div className="login-svgs">
          <DotLottieReact
            src="https://lottie.host/0f263f9b-8666-4863-915f-b65c5b9efd4a/HNPBabEUIZ.lottie"
            loop
            autoplay
          />
        </div>
        <div className="all-login-input-group">
          <img src="" alt="" />
          <label className="all-login-label">Email:</label>
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
          <label className="all-login-label">Password:</label>
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
      </form>
    </div>
  );
};

export default Login;