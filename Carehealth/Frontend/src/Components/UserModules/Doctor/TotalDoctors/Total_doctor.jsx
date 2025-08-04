import React, { useEffect, useState } from "react";
import { toast, ToastContainer } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";
import "./Total_doctor.css";

const TotalDoctor = () => {
  const [doctors, setDoctors] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch("http://localhost:5000/api/patients/totaldoctors")
      .then((response) => response.json())
      .then((data) => {
        setDoctors(data);
        setLoading(false);
        toast.success("Doctors loaded successfully!");
      })
      .catch((error) => {
        console.error("Error fetching doctors:", error);
        toast.error("Error fetching doctors!");
        setLoading(false);
      });
  }, []);

  return (
    <div className="Total-doctor-container">
      <ToastContainer position="top-right" autoClose={3000} />
      <h1 className="Total-doctor-title">Total Doctors</h1>

      {loading ? (
        <p>Loading doctors...</p>
      ) : doctors.length === 0 ? (
        <p>No doctors found.</p>
      ) : (
        <table className="Total-doctor-table">
          <thead>
            <tr className="Total-doctor-table-header">
              <th className="Total-doctor-table-cell">Name</th>
              <th className="Total-doctor-table-cell">Contact</th>
              <th className="Total-doctor-table-cell">Email</th>
              <th className="Total-doctor-table-cell">Specialization</th>
              <th className="Total-doctor-table-cell">Experience</th>
              <th className="Total-doctor-table-cell">Qualification</th>
            </tr>
          </thead>
          <tbody>
            {doctors.map((doctor) => (
              <tr key={doctor._id} className="Total-doctor-table-row">
                <td className="Total-doctor-table-cell">{doctor.name}</td>
                <td className="Total-doctor-table-cell">{doctor.contact}</td>
                <td className="Total-doctor-table-cell">{doctor.email}</td>
                <td className="Total-doctor-table-cell">{doctor.specialization}</td>
                <td className="Total-doctor-table-cell">{doctor.experience}</td>
                <td className="Total-doctor-table-cell">{doctor.qualification}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
};

export default TotalDoctor;
