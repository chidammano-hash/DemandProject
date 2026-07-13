import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { AuthGate } from "../AuthGate";
import { useAuth } from "../../context/AuthContext";

vi.mock("../../context/AuthContext", () => ({ useAuth: vi.fn() }));

const mockedUseAuth = vi.mocked(useAuth);

describe("AuthGate", () => {
  const login = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
    mockedUseAuth.mockReturnValue({ user: null, loading: false, login, logout: vi.fn() });
  });

  it("collects credentials and starts an authenticated session", async () => {
    login.mockResolvedValue(undefined);
    render(<AuthGate><div>Forecast workspace</div></AuthGate>);

    fireEvent.change(screen.getByLabelText("Email"), { target: { value: "planner@example.com" } });
    fireEvent.change(screen.getByLabelText("Password"), { target: { value: "safe-password" } });
    fireEvent.click(screen.getByRole("button", { name: "Sign in" }));

    await waitFor(() => expect(login).toHaveBeenCalledWith({
      email: "planner@example.com",
      password: "safe-password",
    }));
  });

  it("renders the product only after authentication", () => {
    mockedUseAuth.mockReturnValue({
      user: { user_id: "1", email: "planner@example.com", display_name: "Planner", role: "planner" },
      loading: false,
      login,
      logout: vi.fn(),
    });

    render(<AuthGate><div>Forecast workspace</div></AuthGate>);

    expect(screen.getByText("Forecast workspace")).toBeInTheDocument();
  });
});
