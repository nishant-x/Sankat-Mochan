import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import axios from "axios";
import { ToastContainer, toast } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";
import "./DoctorRegistration.css";

const DoctorRegistration = () => {
  const [formData, setFormData] = useState({
    name: "",
    contact: "",
    email: "",
    password: "",
    confirm_Password: "",
    specialization: "",
    experience: "",
    qualification: "",
  });

  const [error, setError] = useState("");
  const navigate = useNavigate();

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    console.log("Submitting form:", formData);

    if (formData.password !== formData.confirm_Password) {
      setError("Passwords do not match!");
      toast.error("Passwords do not match!");
      return;
    }

    try {
      const response = await axios.post("http://localhost:5000/api/register/doctor_register", formData);
      console.log("Response received:", response.data);
      toast.success(response.data);
      navigate("/doctor-login");
    } catch (error) {
      const errMsg = error.response?.data?.message || "Registration failed. Try again.";
      setError(errMsg);
      toast.error(errMsg);
      console.error("Error details:", error.response?.data);
    }
  };

  return (
    <div className="doctor-registration-container">
      <ToastContainer />
      <h1 className="doctor-registration-title">Doctor Register</h1>
      <form className="doctor-registration-form" onSubmit={handleSubmit}>
        <label className="doctor-registration-label" htmlFor="name">
          Full Name:
        </label>
        <input
          className="doctor-registration-input"
          type="text"
          id="name"
          name="name"
          placeholder="Enter your full name"
          value={formData.name}
          onChange={handleChange}
          required
        />

        <label className="doctor-registration-label" htmlFor="contact">
          Contact Number:
        </label>
        <input
          className="doctor-registration-input"
          type="tel"
          id="contact"
          name="contact"
          placeholder="Enter your contact number"
          value={formData.contact}
          onChange={handleChange}
          required
        />

        <label className="doctor-registration-label" htmlFor="email">
          Email:
        </label>
        <input
          className="doctor-registration-input"
          type="email"
          id="email"
          name="email"
          placeholder="Enter your email address"
          value={formData.email}
          onChange={handleChange}
          required
        />

        <label className="doctor-registration-label" htmlFor="password">
          Password:
        </label>
        <input
          className="doctor-registration-input"
          type="password"
          id="password"
          name="password"
          placeholder="Enter your password"
          value={formData.password}
          onChange={handleChange}
          required
        />

        <label className="doctor-registration-label" htmlFor="confirmPassword">
          Confirm Password:
        </label>
        <input
          className="doctor-registration-input"
          type="password"
          id="confirm_Password"
          name="confirm_Password"
          placeholder="Re-enter your password"
          value={formData.confirm_Password}
          onChange={handleChange}
          required
        />

        <label className="doctor-registration-label" htmlFor="specialization">
          Specialization:
        </label>
        <select
          className="doctor-registration-input"
          id="specialization"
          name="specialization"
          value={formData.specialization}
          onChange={handleChange}
          required
        >
          <option value="">Select Department</option>
          <option value="Cardiology">Cardiology</option>
          <option value="Neurology">Neurology</option>
          <option value="Orthopedics">Orthopedics</option>
          <option value="Pediatrics">Pediatrics</option>
          <option value="General Medicine">General Medicine</option>
        </select>

        <label className="doctor-registration-label" htmlFor="experience">
          Years of Experience:
        </label>
        <input
          className="doctor-registration-input"
          type="number"
          id="experience"
          name="experience"
          placeholder="Enter your years of experience"
          value={formData.experience}
          onChange={handleChange}
          required
        />

        <label className="doctor-registration-label" htmlFor="qualification">
          Qualification:
        </label>
        <input
          className="doctor-registration-input"
          type="text"
          id="qualification"
          name="qualification"
          placeholder="Enter your highest qualification"
          value={formData.qualification}
          onChange={handleChange}
          required
        />

        <button className="doctor-registration-button" type="submit">
          Register
        </button>
      </form>
      <p className="doctor-registration-login-text">
        Already have an account? <a href="login.html">Login</a>
      </p>
    </div>
  );
};

export default DoctorRegistration;
