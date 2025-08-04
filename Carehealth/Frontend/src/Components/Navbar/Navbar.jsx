import React, { useContext } from 'react';
import { Link } from 'react-router-dom';
import { useNavigate } from "react-router-dom";
import { FaHome, FaClipboardList, FaCalendarPlus, FaUsers, FaSignOutAlt, FaCog, FaStethoscope } from 'react-icons/fa';
import UserContext from '../Context/UserContext'; // Import the context
import './Navbar.css';

const Navbar = ({ userRole }) => {
  const navigate = useNavigate();
  const { userID } = useContext(UserContext); // Access userID from context
  const renderAdminLinks = () => (
    <>
      <li>
      <a href="#" onClick={() => navigate(-1)}>
          <FaHome size={20} />
          Home
        </a>
      </li>
      <li>
        <a href="#" onClick={() => navigate(-1)}>
        <FaCog size={20}/>
          Admin Dashboard
        </a>
      </li>
      <li>
        <a href="/manage-users">
          <FaUsers size={20} />
          Manage Users
        </a>
      </li>
      <li>
        <a href="/">
          <FaSignOutAlt size={20} />
          Logout
        </a>
      </li>
    </>
  );

  const renderPharmacistLinks = () => (
    <>
      <li>
      <a href="#" onClick={() => navigate(-1)}>
          <FaHome size={20} />
          Home
        </a>
      </li>
      <li>
        <a href="/inventory">
          <FaClipboardList size={20} />
          Manage Medicines
        </a>
      </li>
      <li>
        <a href="/">
          <FaSignOutAlt size={20} />
          Logout
        </a>
      </li>
    </>
  );

  const renderPatientLinks = () => (
    <>
      <li>
        <a href="#" onClick={() => navigate(-1)}>
          <FaHome size={20} />
          Home
        </a>
      </li>
      <li>
      <Link to={`/patient/appoinments`} state={{ userDetails: userID }}>
          <FaClipboardList size={20} />
          My Appointments
          </Link>
      </li>
      <li>
      <Link to={`/patient/prescription/${userID}`} state={{ userDetails: userID }}>
        <FaCalendarPlus size={20} />
         My Prescriptions
        </Link>
  
      </li>
      <li>
        <a href="/">
          <FaSignOutAlt size={20} />
          Logout
        </a>
      </li>
    </>
  );

  const renderDoctorLinks = () => (
    <>
      <li>
      <a href="#" onClick={() => navigate(-1)}>
          <FaHome size={20} />
          Home
        </a>
      </li>
      <li>
        <a href="#">
          <FaClipboardList size={20} />
          Manage Appointments
        </a>
      </li>
      <li>
        <a href="/">
          <FaSignOutAlt size={20} />
          Logout
        </a>
      </li>
    </>
  );

  const renderGuestLinks = () => (
    <>
      <li>
        <a href="/">
          <FaHome size={20} />
          Home
        </a>
      </li>
      <li>
        <a href="/login">
          <FaSignOutAlt size={20} />
          Login
        </a>
      </li>
    </>
  );



  return (
    <nav className="navbar">
      <div className="navbar-brand">
        <h2>OPD Management</h2>
      </div>
      <ul className="navbar-links">
        {userRole === 'guest' && renderGuestLinks()}
        {userRole === 'admin' && renderAdminLinks()}
        {userRole === 'pharmacist' && renderPharmacistLinks()}
        {userRole === 'patient' && renderPatientLinks()}
        {userRole === 'doctor' && renderDoctorLinks()}
      </ul>
    </nav>
  );
};

export default Navbar;