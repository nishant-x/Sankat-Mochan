import React, { useEffect, useState } from "react";
import { toast } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";
import "./Total_patient.css";

const TotalPatient = () => {
    const [patients, setPatients] = useState([]);
    const [loading, setLoading] = useState(true); // Added loading state

    useEffect(() => {
        fetch("http://localhost:5000/api/patients/totalpatients")
            .then((response) => response.json())
            .then((data) => {
                setPatients(data);
                setLoading(false);
            })
            .catch((error) => {
                console.error("Error fetching patients:", error);
                toast.error("Failed to load patient data!");
                setLoading(false);
            });
    }, []);

    return (
        <div className="Total-patient-container">
            <h1 className="Total-patient-title">Total Patients</h1>

            {loading ? ( // Show loading message while fetching data
                <p className="Total-patient-loading">Loading patient data...</p>
            ) : patients.length > 0 ? (
                <table className="Total-patient-table">
                    <thead>
                        <tr className="Total-patient-header-row">
                            <th className="Total-patient-header">Name</th>
                            <th className="Total-patient-header">Age</th>
                            <th className="Total-patient-header">Gender</th>
                            <th className="Total-patient-header">Contact No.</th>
                            <th className="Total-patient-header">Email</th>
                            <th className="Total-patient-header">Blood Group</th>
                            <th className="Total-patient-header">Address</th>
                        </tr>
                    </thead>
                    <tbody>
                        {patients.map((patient) => (
                            <tr key={patient._id} className="Total-patient-row">
                                <td className="Total-patient-cell">{patient.name}</td>
                                <td className="Total-patient-cell">{patient.age}</td>
                                <td className="Total-patient-cell">{patient.gender}</td>
                                <td className="Total-patient-cell">{patient.contact}</td>
                                <td className="Total-patient-cell">{patient.email}</td>
                                <td className="Total-patient-cell">{patient.bloodGroup}</td>
                                <td className="Total-patient-cell">{patient.address}</td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            ) : (
                <p className="Total-patient-no-data">No patients found.</p>
            )}
        </div>
    );
};

export default TotalPatient;
