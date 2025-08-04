import React from 'react';
import './Footer.css';
import {
  FaMapMarkerAlt,
  FaPhone,
  FaEnvelope,
  FaFacebookF,
  FaTwitter,
  FaInstagram,
  FaLinkedinIn,
  FaHospital,
  FaBed,
  FaCalendarCheck,
  FaUserMd,
  FaHome,          // Added for Home icon
  FaInfoCircle,    // Added for About Us icon
  FaMedkit,        // Added for Services icon
} from 'react-icons/fa';

export default function Footer() {
  return (
    <footer className="footer-container">
      <div className="footer-content">
        {/* About Section */}
        <div className="footer-section">
          <h3 className="footer-heading">About Us</h3>
          <p className="footer-text">
            Our Hospital Management System is designed to streamline hospital operations, including inventory management, bed allocation, and appointment scheduling. We aim to provide efficient and patient-centric healthcare solutions.
          </p>
        </div>

        {/* Quick Links Section */}
        <div className="footer-section">
          <h3 className="footer-heading">Quick Links</h3>
          <ul className="footer-links">
            <li>
              <a href="#" className="footer-link">
                <FaHome /> Home
              </a>
            </li>
            <li>
              <a href="#" className="footer-link">
                <FaInfoCircle /> About Us
              </a>
            </li>
            <li>
              <a href="#" className="footer-link">
                <FaMedkit /> Services
              </a>
            </li>
            <li>
              <a href="#" className="footer-link">
                <FaEnvelope /> Contact Us
              </a>
            </li>
          </ul>
        </div>

        {/* Contact Section */}
        <div className="footer-section">
          <h3 className="footer-heading">Contact Us</h3>
          <ul className="footer-contact">
            <li>
              <FaMapMarkerAlt className="footer-icon" /> 123 Health Street, MedCity, Country
            </li>
            <li>
              <FaPhone className="footer-icon" /> +123 456 7890 (Emergency)
            </li>
            <li>
              <FaPhone className="footer-icon" /> +123 456 7891 (Appointments)
            </li>
            <li>
              <FaEnvelope className="footer-icon" /> info@hospitalmanagement.com
            </li>
          </ul>
        </div>

        {/* Services Section */}
        <div className="footer-section">
          <h3 className="footer-heading">Our Services</h3>
          <ul className="footer-links">
            <li><a href="#" className="footer-link">Patient Registration</a></li>
            <li><a href="#" className="footer-link">Electronic Health Records (EHR)</a></li>
            <li><a href="#" className="footer-link">Pharmacy Management</a></li>
            <li><a href="#" className="footer-link">Laboratory Management</a></li>
            <li><a href="#" className="footer-link">Billing & Invoicing</a></li>
          </ul>
        </div>
      </div>

      {/* Footer Bottom */}
      <div className="footer-bottom">
        <p className="footer-copy">&copy; 2024 Hospital Management System. All Rights Reserved.</p>
        <div className="footer-social">
          <a href="#" className="social-icon">
            <FaFacebookF />
          </a>
          <a href="#" className="social-icon">
            <FaTwitter />
          </a>
          <a href="#" className="social-icon">
            <FaInstagram />
          </a>
          <a href="#" className="social-icon">
            <FaLinkedinIn />
          </a>
        </div>
      </div>
    </footer>
  );
}