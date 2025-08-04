import React, { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { DotLottieReact } from "@lottiefiles/dotlottie-react";
import axios from "axios";
import { ToastContainer, toast } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";
import '../AllLogin.css';

export default function PatientLogin() {
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
      const response = await axios.post("http://localhost:5000/api/login/patient_login", formData);

      // Assuming 'response.data' contains the user details
      const userDetails = response.data;
      localStorage.setItem("userDetails", JSON.stringify(userDetails));
      console.log(`User Details: ${JSON.stringify(userDetails)}`);

      // Show success toast
      toast.success("Login Successful!", {
        position: "top-right",
        autoClose: 3000,
        hideProgressBar: false,
        closeOnClick: true,
        pauseOnHover: true,
        draggable: true,
      });

      // Pass the response data to the next page using navigate
      navigate("/patient-dashboard", { state: { userDetails } }); // Passing user details as state
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

      <h1>Patient Login</h1>
      <form onSubmit={handleSubmit}>
        <DotLottieReact
          src="https://lottie.host/5d84c14b-8935-40f2-a8a7-2996b13dec61/PMwIOzHema.lottie"
          loop
          autoplay
        />
        {error && <p className="all-login-error">{error}</p>}

        <div className="all-login-input-group">
          <label>Email:</label>
          <input 
            type="email" 
            name="email" 
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
            value={formData.password} 
            onChange={handleChange} 
            required 
          />
        </div>

        <button type="submit" className="all-login-button">Login</button>

        <p className="all-login-register-link">
          Don't have an account? <Link to="/patient-registration">Register Now</Link>
        </p>
      </form>
    </div>
  );
}