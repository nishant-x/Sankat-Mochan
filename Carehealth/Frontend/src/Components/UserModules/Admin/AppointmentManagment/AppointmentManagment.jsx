import React from "react";
import { ToastContainer, toast } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";
import './AppointmentManagment.css';
import { FaCalendarPlus, FaClipboardList, FaClock, FaUsers, FaRegCheckCircle } from 'react-icons/fa';

export default function QueueManagement() {
  // Example handler to trigger a toast notification
  const handleClick = (msg) => {
    toast.info(msg);
  };

  return (
    <div>
      <ToastContainer />
      <h1 className="AppointmentManagement-heading">Appointment Management</h1>
      <div className="AppointmentManagement-container">
        {/* Add New Appointment Card */}
        <div 
          className="AppointmentManagement-card" 
          onClick={() => handleClick("Add New Appointment clicked")}
        >
          <FaCalendarPlus className="AppointmentManagement-icon" />
          <h2 className="AppointmentManagement-card-title">Add New Appointment</h2>
        </div>

        {/* Today's Appointments Card */}
        <div 
          className="AppointmentManagement-card" 
          onClick={() => handleClick("Today's Appointments clicked")}
        >
          <FaClipboardList className="AppointmentManagement-icon" />
          <h2 className="AppointmentManagement-card-title">Today's Appointments</h2>
          <p className="AppointmentManagement-card-description">5 appointments remaining</p>
        </div>

        {/* Tomorrow's Appointments Card */}
        <div 
          className="AppointmentManagement-card" 
          onClick={() => handleClick("Tomorrow's Appointments clicked")}
        >
          <FaClock className="AppointmentManagement-icon" />
          <h2 className="AppointmentManagement-card-title">Tomorrow's Appointments</h2>
          <p className="AppointmentManagement-card-description">8 appointments scheduled</p>
        </div>

        {/* Waiting for Appointments Card */}
        <div 
          className="AppointmentManagement-card" 
          onClick={() => handleClick("Waiting for Appointments clicked")}
        >
          <FaClipboardList className="AppointmentManagement-icon" />
          <h2 className="AppointmentManagement-card-title">Waiting for Appointments</h2>
          <p className="AppointmentManagement-card-description">2 patients in the queue</p>
        </div>

        {/* Upcoming Appointments Card */}
        <div 
          className="AppointmentManagement-card" 
          onClick={() => handleClick("Upcoming Appointments clicked")}
        >
          <FaUsers className="AppointmentManagement-icon" />
          <h2 className="AppointmentManagement-card-title">Upcoming Appointments</h2>
          <p className="AppointmentManagement-card-description">15 upcoming appointments</p>
        </div>

        {/* Past Appointments Card */}
        <div 
          className="AppointmentManagement-card" 
          onClick={() => handleClick("Past Appointments clicked")}
        >
          <FaRegCheckCircle className="AppointmentManagement-icon" />
          <h2 className="AppointmentManagement-card-title">Past Appointments</h2>
          <p className="AppointmentManagement-card-description">10 completed appointments</p>
        </div>
      </div>
    </div>
  );
}
