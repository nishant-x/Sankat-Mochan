import React, { useEffect, useState, useContext } from "react";
import { useLocation, Link, useNavigate } from "react-router-dom";
import axios from "axios";
import { toast } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";
import BedStatus from "../../../BedManagement/BedStatus/BedStatus.jsx";
import UserContext from "../../../Context/UserContext.js";
import "./DoctorDashboard.css";



const DoctorDashboard = () => {
  const [doctor, setDoctor] = useState(null);
  const [patients, setPatients] = useState([]);
  const { userID, setUserID } = useContext(UserContext);
  const [loading, setLoading] = useState(true);
  const location = useLocation();
  const { userDetails } = location.state || {};
  const navigate = useNavigate();

  useEffect(() => {
    const fetchDoctorAndPatients = async () => {
      if (!userDetails?.doctor?._id) return;
      try {
        setUserID(userDetails.doctor._id);
        const response = await axios.get(
          `http://localhost:5000/doctorRoutes/doctor_patient_list/${userDetails.doctor._id}`
        );
        setDoctor(response.data.doctor);
        setPatients(response.data.patients);
      } catch (error) {
        console.error("Error fetching data:", error);
        toast.error("Error fetching data.");
      } finally {
        setLoading(false);
      }
    };
    fetchDoctorAndPatients();
  }, [userDetails?.doctor?._id, setUserID]);

  const handleSubmit = async (patientId) => {
    try {
      await axios.put("http://localhost:5000/doctorRoutes/doctor_patient_list", {
        patientId,
        reached: true,
      });
      setPatients((prevPatients) =>
        prevPatients.map((patient) =>
          patient._id === patientId ? { ...patient, reached: true } : patient
        )
      );
      toast.success("Patient status updated successfully!");
    } catch (error) {
      console.error("Error updating patient status:", error);
      toast.error("Failed to update patient status.");
    }
  };

  const handleInventoryRequest = () => {
    navigate("/inventory-request", { state: { doctor_id: userDetails.doctor._id } });
  };

  return (
    <div className="doctor-dash-container">
      <h2 className="doctor-dash-title">Doctor Dashboard</h2>

      {loading ? (
        <p className="doctor-dash-loading-message">Loading doctor and patient details...</p>
      ) : doctor ? (
        <div className="doctor-dash-info">
          <h3 className="doctor-dash-name">Dr. {doctor.name}</h3>
          <p><strong>Specialization:</strong> {doctor.specialization}</p>
          <p><strong>Email:</strong> {doctor.email}</p>
          <p><strong>Phone:</strong> {doctor.contact}</p>
        </div>
      ) : (
        <p className="doctor-dash-not-found-message">Doctor not found.</p>
      )}

      <h3 className="doctor-dash-subtitle">
        Request to Inventory
        <button onClick={handleInventoryRequest} className="doctor-dash-bed-btn">Request Inventory</button>
      </h3>

      <h3 className="doctor-dash-subtitle">Appointments</h3>
      {patients.length > 0 ? (
        <table className="doctor-dash-table">
          <thead>
            <tr>
              <th>Patient Name</th>
              <th>Department</th>
              <th>Date</th>
              <th>Time</th>
              <th>Add Prescription</th>
              <th>View Prescription</th>
              <th>Allocate Bed</th>
              <th>Prescription Status</th>
            </tr>
          </thead>
          <tbody>
            {patients.map((patient) => (
              <tr key={patient._id}>
                <td>{patient.name}</td>
                <td>{patient.department}</td>
                <td>{patient.date}</td>
                <td>{patient.time}</td>
                <td>
                  <Link to="/add-patient-med" state={{ patient, doctor }} className="doctor-dash-add-btn">ADD</Link>
                </td>
                <td>
                  <Link to={`/patient/prescription/${patient.patientId}`} state={{ userDetails: patient }} className="doctor-dash-view-btn">View Prev Prescription</Link>
                </td>
                <td>
                  <Link to="/bed-application" state={{ patient, doctor }} className="doctor-dash-bed-btn">ADD</Link>
                </td>
                <td>
                  <button onClick={() => handleSubmit(patient._id)} disabled={patient.reached} className="doctor-dash-bed-btn">
                    {patient.reached ? "Done" : "Pending..."}
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <p className="doctor-dash-no-patients-message">No patients found in this department.</p>
      )}

      <h3 className="doctor-dash-subtitle">Beds Allocated to Doctor</h3>
      {userDetails?.doctor?._id ? (
        <BedStatus doctor={userDetails.doctor} />
      ) : (
        <p>Loading doctor data...</p>
      )}
    </div>
  );
};

export default DoctorDashboard;