import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import axios from "axios";
import { ToastContainer, toast } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";
import './PharmacyRegistration.css';

const PharmacyRegistration = () => {
  const [formData, setFormData] = useState({
    name: "",
    contact: "",
    email: "",
    password: "",
    confirm_password: "", // Ensure correct naming
    department: "",
    yearsOfExperience: "",
  });

  const [error, setError] = useState("");
  const navigate = useNavigate();

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData({
      ...formData,
      [name]: name === "yearsOfExperience" ? (value ? Number(value) : "") : value,
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(""); // Clear previous error

    if (formData.password !== formData.confirm_password) {
      setError("Passwords do not match!");
      toast.error("Passwords do not match!");
      return;
    }

    // Ensure yearsOfExperience is not empty
    if (!formData.yearsOfExperience) {
      setError("Years of Experience is required.");
      toast.error("Years of Experience is required.");
      return;
    }

    try {
      const payload = {
        ...formData,
        years_of_experience: Number(formData.yearsOfExperience), // Match backend naming
      };

      const response = await axios.post("http://localhost:5000/api/register/inventory_register", payload);
      toast.success(response.data.message || "Registration Successful!");
      navigate("/pharmacy-login");
    } catch (error) {
      const errMsg = error.response?.data?.message || "Registration failed. Try again.";
      setError(errMsg);
      toast.error(errMsg);
      console.error("Error details:", error.response?.data);
    }
  };

  return (
    <div className="inventory-manager-register-container">
      <ToastContainer />
      <h1 className="inventory-manager-register-title">Inventory Manager Register</h1>
      {error && <p className="error-message">{error}</p>}
      <form className="inventory-manager-register-form" onSubmit={handleSubmit}>
        <label htmlFor="name">Full Name:</label>
        <input
          type="text"
          id="name"
          name="name"
          placeholder="Enter your full name"
          value={formData.name}
          onChange={handleChange}
          required
        />

        <label htmlFor="contact">Contact Number:</label>
        <input
          type="tel"
          id="contact"
          name="contact"
          placeholder="Enter your contact number"
          value={formData.contact}
          onChange={handleChange}
          required
        />

        <label htmlFor="email">Email:</label>
        <input
          type="email"
          id="email"
          name="email"
          placeholder="Enter your email address"
          value={formData.email}
          onChange={handleChange}
          required
        />

        <label htmlFor="password">Password:</label>
        <input
          type="password"
          id="password"
          name="password"
          placeholder="Enter your password"
          value={formData.password}
          onChange={handleChange}
          required
        />

        <label htmlFor="confirm_password">Confirm Password:</label>
        <input
          type="password"
          id="confirm_password"
          name="confirm_password"
          placeholder="Re-enter your password"
          value={formData.confirm_password}
          onChange={handleChange}
          required
        />

        <label htmlFor="department">Department:</label>
        <input
          type="text"
          id="department"
          name="department"
          placeholder="Enter your department"
          value={formData.department}
          onChange={handleChange}
          required
        />

        <label htmlFor="yearsOfExperience">Years of Experience:</label>
        <input
          type="number"
          id="yearsOfExperience"
          name="yearsOfExperience"
          placeholder="Enter your years of experience"
          value={formData.yearsOfExperience}
          onChange={handleChange}
          required
        />

        <button type="submit" className="inventory-manager-register-button">
          Register
        </button>
      </form>
      <p className="inventory-manager-register-login-text">
        Already have an account? <a href="/pharmacy-login">Login</a>
      </p>
    </div>
  );
};

export default PharmacyRegistration;
