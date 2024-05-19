import Container from "react-bootstrap/Container";
import Nav from "react-bootstrap/Nav";
import Navbar from "react-bootstrap/Navbar";

import { Link } from "react-router-dom";

const NavbarComponent = () => {
  return (
    <Navbar bg="dark" data-bs-theme="dark">
      <Container>
        <Nav className="me-auto gap-3 d-flex">
          <Link to={"/"}>Lowpass</Link>
          <Link to={"/butterworth"}>Butterworth</Link>
          <Link to={"/histogram"}>Histogram Matching</Link>
          <Link to={"/highpass"}>HighPass</Link>
          <Link to={"/feature"}>SIFT Feature</Link>
        </Nav>
      </Container>
    </Navbar>
  );
};

export default NavbarComponent;
