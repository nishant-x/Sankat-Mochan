import React from 'react';
import { Link } from 'react-router-dom';
import { DotLottieReact } from "@lottiefiles/dotlottie-react";
import { toast } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";
import './PharmacyDashboard.css';

const LandingPage = () => {
    const handleNavigation = (message) => {
        toast.info(message, { position: "top-right", autoClose: 2000 });
    };

    return (
        <div className="Pharmacydash">
            <div className="pharmacy-dash-container">
                <h1 className="pharmacy-dash-heading">Welcome, Pharmacist</h1>
                <div className="pharmacy-dash-options-container">

                    <Link to="/patient-list" className="pharmacy-dash-option-link" onClick={() => handleNavigation("Navigating to Patient List")}>
                        <DotLottieReact
                            src="https://lottie.host/0f263f9b-8666-4863-915f-b65c5b9efd4a/HNPBabEUIZ.lottie"
                            loop
                            autoplay
                        />
                        <br />
                        <span>View Patient List</span>
                    </Link>

                    <Link to="/inventory" className="pharmacy-dash-option-link" onClick={() => handleNavigation("Navigating to Inventory")}>
                        <DotLottieReact
                            src="https://lottie.host/0f263f9b-8666-4863-915f-b65c5b9efd4a/HNPBabEUIZ.lottie"
                            loop
                            autoplay
                        />
                        <br />
                        <span>Check Inventory</span>
                    </Link>

                    <Link to="/hospital-demands" className="pharmacy-dash-option-link" onClick={() => handleNavigation("Navigating to Hospital Demands")}>
                        <DotLottieReact
                            src="https://lottie.host/0f263f9b-8666-4863-915f-b65c5b9efd4a/HNPBabEUIZ.lottie"
                            loop
                            autoplay
                        />
                        <br />
                        <span>Manage Hospital Demands</span>
                    </Link>

                </div>
            </div>
        </div>
    );
};

export default LandingPage;
