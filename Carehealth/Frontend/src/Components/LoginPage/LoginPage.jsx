import React from "react";
import { Link } from "react-router-dom";
// import { TbEmergencyBed } from "react-icons/tb";
import { MdEmergency } from "react-icons/md";
import { FaHandHoldingMedical } from "react-icons/fa6";
import { FaHeartbeat } from "react-icons/fa";
import { FaArrowRightLong } from "react-icons/fa6";
import docter from "../../assets/doctor.jpg";
// import Slider from '..//slider/slider';
import { DotLottieReact } from "@lottiefiles/dotlottie-react";
import Imageslide from "../Slider/Imageside.jsx";
import myIcon from "../../assets/docter.svg";
import "./LoginPage.css";
import {
  FaUserCog,
  FaStethoscope,
  FaUserMd,
  FaPrescriptionBottleAlt,
} from "react-icons/fa";

// import video from "../../assets/doctor.mp4";
// import "./LoginPage.css";

export default function HomePage() {
  return (
    <div className="homepage-container">
      {/* <Slider images={images}/> */}
      <Imageslide />
      <div className="parent">
        <div className="box1">
          <h2 id="h2">24 Hours service</h2>
          <div className="child-box1">
            <p id="p">
              6 This idea ties into the idea of 24/7 service, where time is
              always ticking, but theres always an opportunity to do something
              impactful or positive.
            </p>
          </div>
          <button className="read">Read more</button>
        </div>
        <div className="box2">
          <h2 id="h2">Happy with you smile</h2>
          <div className="child-box2">
            <p id="p">
              Cum sociis natoque penatibus et magnis dis parturient montesmus.
              Pro vel nibh et elit mollis commodo et nec augue tristique sed
              volutpat.
            </p>
          </div>
          <button className="read">Read more</button>
        </div>
        <div className="box3">
          <h2 id="h2">opening hours</h2>
          <div className="child-box3">
            <p id="p">
              Time 8 AM to 10 PM <br />
              Monday <br />
              Tuesday
            </p>
            <button className="read">Read more</button>
          </div>
        </div>
      </div>

      {/* Login Section */}
      {/* <h2 className="section-title">Login</h2> */}
      <div className="login-type-container">
        <Link to="/admin-login" className="login-card">
        <DotLottieReact
      src="https://lottie.host/0f263f9b-8666-4863-915f-b65c5b9efd4a/HNPBabEUIZ.lottie"
      loop
      autoplay
          />
          <span className="login-span">Admin Login</span>
        </Link>
        <Link to="/patient-login" className="login-card">
        <DotLottieReact
      src="https://lottie.host/5d84c14b-8935-40f2-a8a7-2996b13dec61/PMwIOzHema.lottie"
      loop
      autoplay
    />
          <span className="login-span">Patient Login</span>
        </Link>
        <Link to="/doctor-login" className="login-card">
          <DotLottieReact
            src="https://lottie.host/f79a6b3d-f9c6-4c33-9323-bef8a558e756/mhKDzhTWrt.lottie"
            loop
            autoplay
          />
          <span className="login-span">Doctor Login</span>
        </Link>
        <Link to="/pharmacy-login" className="login-card">
        <DotLottieReact
      src="https://lottie.host/e088efab-c64c-4256-9cbd-0cc53bc83a2a/ewP7msP3mq.lottie"
      loop
      autoplay
    />
          <span className="login-span">Pharmacy Login</span>
        </Link>
      </div>

      <div className="information">
        <div className="inbox">
        <MdEmergency id="icons" size={50} style={{ color: "red" }} />
          <br />
          <h4 id="h2">Emergency</h4>
          <br />
          One of the key classifications is between traditio nal small molecule
          drugs; usually derived from chemical synthesis.
          <button className="lern-btn">
            {" "}
            learn more <FaArrowRightLong />
          </button>
        </div>

        <div className="inbox">
          <FaHeartbeat id="icons" size={50} style={{ color: "red" }}/>
          <br />
          <h4 id="h2">Heart Diseases</h4>
          <br />
          One of the key classifications is between traditio nal small molecule
          drugs; usually derived from chemical synthesis.
          <button className="lern-btn">
            {" "}
            learn more <FaArrowRightLong />
          </button>
        </div>
        <div className="inbox">
          <FaHandHoldingMedical id="icons" size={50} style={{ color: "red" }}/>
          <br />
          <h4 id="h2">Orthopaedic</h4>
          <br />
          One of the key classifications is between traditio nal small molecule
          drugs; usually derived from chemical synthesis.
          <button className="lern-btn">
            {" "}
            learn more <FaArrowRightLong />
          </button>
        </div>
      </div>
    </div>
  );
}