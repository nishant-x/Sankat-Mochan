import React, { useContext, useEffect, useState } from "react";
import { Link, useNavigate, useLocation } from "react-router-dom";
import Chatbot from "../../../Chatbot/Chatbot";
import UserContext from "../../../Context/UserContext";
import "./PatientDashboard.css";
// import "./patient-info-card.css";
import { IoClose } from "react-icons/io5";
import PatientPrescrptionDashBoard from "../PatientPrescription/PatientPrescriptionDashBoard.jsx";
import RescheduleAppoinment from "../../../OPDQ/AppointmentsList.jsx";

const PatientDashboard = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const { userDetails } = location.state || {};
  const { userID, setUserID } = useContext(UserContext);
  const [isChatbotOpen, setIsChatbotOpen] = useState(false);
  const [totalAppointment, setTotalAppointment] = useState(null);
  const [latestAppointment, setLatestAppointment] = useState(null);

  const fetchtxtfile = async () => {
    try {
      if (!userDetails) {
        console.error("User ID not found.");
        return;
      } else {
        setUserID(userDetails.patient._id);
      }

      const response = await fetch(
        `http://localhost:5000/api/login/fetchtxtfile/${userDetails.patient._id}`
      );

      if (!response.ok) {
        throw new Error("Failed to fetch the file");
      }
      const textData = await response.text();

      await sendTextToBackend(textData);
    } catch (error) {
      console.error("Error fetching text file:", error);
    }
  };

  const sendTextToBackend = async (textData) => {
    try {
      const response = await fetch("http://localhost:8080/store-text", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ content: textData }),
      });

      if (!response.ok) {
        throw new Error("Failed to send text data to backend");
      }

      const result = await response.json();
      console.log("Text stored successfully:", result);
    } catch (error) {
      console.error("Error sending text data to backend:", error);
    }
  };

  useEffect(() => {
    const fetchAppointmentCount = async () => {
      if (!userDetails?.patient?._id) {
        console.error("Patient ID is missing");
        return;
      }

      try {
        const response = await fetch(
          `http://localhost:5000/api/appointments/patient/total_appointment/${userDetails.patient._id}`
        );
        const data = await response.json();

        if (response.ok) {
          setTotalAppointment(data.total_appointment);
        } else {
          console.error(
            "Error from server:",
            data.message || "Failed to fetch total appointments"
          );
        }
      } catch (err) {
        console.error("Error fetching total appointments:", err);
      }
    };

    fetchAppointmentCount();
  }, [userDetails?.patient?._id]);

  const fetchLatestAppointment = async () => {
    if (!userDetails?.patient?._id) {
      console.error("Patient ID is missing");
      return;
    }

    try {
      console.log("from patient dashboard", userDetails.patient._id);

      const response = await fetch(
        `http://localhost:5000/api/appointments/patient/latest_appointment/${userDetails.patient._id}`
      );
      const data = await response.json();

      if (response.ok) {
        setLatestAppointment(data);
      } else {
        console.error("Error fetching latest appointment:", data.message);
      }
    } catch (err) {
      console.error("Error fetching latest appointment:", err);
    }
  };

  useEffect(() => {
    fetchLatestAppointment();
  }, []);

  useEffect(() => {
    if (userDetails?.patient?._id) {
      setUserID(userDetails.patient._id);
    }
    fetchtxtfile();
    if (!userDetails || !userDetails.patient) {
      navigate("/login");
    }
  }, [userDetails, navigate, setUserID]);

  const toggleChatbot = () => {
    setIsChatbotOpen((prevState) => !prevState);
  };

  return (
    <div className="patient-dash-container">
      <nav className="patient-dash-nav">
        <h1 className="patient-dash-title">Patient Dashboard</h1>
        <Link
          className="patient-dash-link-appointments"
          to="/appointment-booking"
          state={{ userDetails: userDetails.patient }}
        >
          Appointments
        </Link>
        <Link
          className="patient-dash-link-prescription"
          to={`/patient/prescription/${userDetails.patient._id}`}
          state={{ userDetails: userDetails.patient }}
        >
          View Prescription
        </Link>
        <Link
          className="patient-dash-link-bills"
          to="/patient-bill"
          state={{ userDetails: userDetails.patient }}
        >
          Bills
        </Link>
        <Link
          className="patient-dash-link-reschedule"
          to="/appointment-reschedule"
          state={{ userDetails: userDetails.patient }}
        >
          Reschedule Appointment
        </Link>
      </nav>
      <div className="patient-dash-details-cards">
        <div className="patient-dash-details-card-upper">


          
        {/* patient details */}
        <div className="profile-widget-container">
          <div className="profile-header">
            <img
              src="https://th.bing.com/th/id/OIP.dtiVnrbOAPlyB_q_jphv_QHaEK?rs=1&pid=ImgDetMain"
              alt="Profile"
              className="profile-image"
            />
          </div>
          <div className="profile-info">
            <h2 className="profile-name"> {userDetails.patient.name} </h2>
            <p className="profile-role">Email:{userDetails.patient.email}</p>
            <p className="profile-gender">Gende:{userDetails.patient.gender}</p>
            <p className="profile-bloodGroup">BloodGroup:{userDetails.patient.bloodGroup}</p>
            <p className="profile-contact">Contact:{userDetails.patient.contact}</p>
            <p className="profile-bio">Address: {userDetails.patient.address}</p>
          </div>
        </div>
        <div className="upper-patient-right">
          <div className="patient-dashboard-btn">
            {/* total count */}
              <div className="patient-appointment-total-count">
                <h2 className="patient-appointment-total-count-title">
                  Total Appointments:{" "}
                  {totalAppointment !== null ? totalAppointment : "Loading..."}
                </h2>
              </div>
            {/* medicalreport */}
            <div className="medicalreportanalyser">
              <Link
                className="patient-dash-link-medical-report"
                to="/MedicalReportAnalyzer"
                state={{ userDetails }}
              >
                Medical Report Analyzer
              </Link>
            </div>
          </div>
            <div className="patient-dashboard-cards-appointment">
              {/* latest appointment */}
              <div className="latest-appointment-container">
                <h2 className="latest-appointment-title">Upcoming Appointment</h2>

                {latestAppointment ? (
                  <div className="latest-appointment-card">
                    <p>
                      <strong>Date:</strong>{" "}
                      {latestAppointment.date
                        ? new Date(latestAppointment.date).toLocaleDateString(
                            "en-US",
                            {
                              year: "numeric",
                              month: "long",
                              day: "numeric",
                            }
                          )
                        : "N/A"}
                    </p>
                    <p>
                    <strong>Time:</strong>{" "}
                    {latestAppointment.time}

                    </p>
                    <p>
                      <strong>Doctor:</strong>{" "}
                      {latestAppointment.doctorName || "N/A"}
                    </p>
                    <p>
                      <strong>Department:</strong> {latestAppointment.department || "N/A"}
                    </p>
                  </div>
                ) : (
                  <p>No upcoming appointments found.</p>
                )}
              </div>
              {/* this patient prescription */}
              <PatientPrescrptionDashBoard userDetails={userDetails.patient} />
            </div>
        </div>
        </div>
        {/* reschedule */}
        <div className="reschedule">
          <RescheduleAppoinment userDetails={userDetails.patient._id} />
        </div>
      </div>

      <div className="chatbot_logo" onClick={toggleChatbot}>
        <img
          src="https://cdni.iconscout.com/illustration/premium/thumb/online-medical-chatbot-service-5588600-4655407.png"
          alt="Medical Chatbot Icon"
        />
      </div>

      {isChatbotOpen && (
        <div className="chatbot_popup">
          <div className="chatbot_header">
            <button onClick={toggleChatbot} className="close_button">
              <IoClose />
            </button>
          </div>
          <Chatbot userDetails={userDetails} />
        </div>
      )}
    </div>
  );
};

export default PatientDashboard;
