import React from "react";
import "./Imageside.css";
import myIcon from "../../assets/docter.svg";
import { FaArrowRight } from "react-icons/fa";
import { DotLottieReact } from "@lottiefiles/dotlottie-react";
// import Img from '../../assets/templete.jpg';

function imgside() {
  return (
    <div className="imgCantainer">
      {/* <img src={Img} alt="" /> */}
      <div className="imgText">
        <h1 className="text">
          Discover Health: <br /> Find Your Trusted img <span>Doctors</span>{" "}
          Today.
        </h1>
        <div className="svg-homeage">
          <DotLottieReact
            src="https://lottie.host/aefd25a9-0761-4192-8a02-a1c591fc71a2/kAggrIqva7.lottie"
            loop
            autoplay
          />
        </div>

        <div className="dotLot">
          <button className="booking-button">
            Booking Appointment <FaArrowRight className="arrow-icon" />
          </button>
        </div>
      </div>
      <div className="imgbackground">
        <img src={myIcon} alt="doctor" className="img" />
      </div>
    </div>
  );
}

export default imgside;