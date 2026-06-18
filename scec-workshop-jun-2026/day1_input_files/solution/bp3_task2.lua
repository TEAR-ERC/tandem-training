local BP3 = {}
BP3.__index = BP3

-- Constant parameters
BP3.a0 = 0.010      -- a value in velocity-weakening zone
BP3.amax = 0.025    -- a value in velocity-strengthening zone
BP3.b = 0.019       -- Task 2: increased evolution-effect parameter b (was 0.015)
                    -- -> larger (a-b) weakening
BP3.H = 15.0        -- Depth of velocity-weakening zone
BP3.h = 3.0         -- Thickness of transition zone, from velocity-weakening to velocity-strengthening
BP3.V0 = 1.0e-6     -- Reference slip rate [m/s]
BP3.f0 = 0.6        -- Reference friction coefficient
BP3.nu = 0.25       -- Poisson ratio

-- Function to initiate new scenario
function BP3.new(params)
    -- You can define parameters that you may want to change for each scenario
    local self = setmetatable({}, BP3)
    self.dip = params.dip * math.pi / 180.0     -- dip in radians
    self.Vp = params.Vp                         -- convergence rate [m/s]
    return self
end

-- Boundary condition governing Dirichlet boundary in the mesh
function BP3:boundary(x, y, t)
    -- Returns (slip along x-axis, slip along y-axis)
    local Vh = self.Vp * t
    local dist = x + y / math.tan(self.dip)
    if dist > 1 then
        Vh = -Vh / 2.0
    elseif dist < -1 then
        Vh = Vh / 2.0
    end
    return Vh * math.cos(self.dip), -Vh * math.sin(self.dip)
end

-- Shear wave velocity [km/s]; to define material property
function BP3:shearVel(x, y)
    return 4.0
end

-- Density [g/cm3]; to define material property
function BP3:density(x, y)
    return 2.9
end

-- Shear modulus [GPa]
function BP3:mu(x, y)
    return self:shearVel(x, y)^2 * self:density(x, y)
end

-- Lame parameter lambda [GPa]
function BP3:lam(x, y)
    return 2 * self.nu * self:mu(x, y) / (1 - 2 * self.nu)
end

-- Half the shear-wave impedance for radiation damping [MPa s/m]
function BP3:eta(x, y)
    return self:shearVel(x, y) * self:density(x, y) / 2.0
end

-- Critical distance or state evolution distance (D_c) [m]
function BP3:L(x, y)
    -- Task 2: increased critical slip distance D_c (was 0.008 -> 0.012 m).
    -- Larger D_c grows the nucleation size and lengthens the recurrence interval.
    return 0.012
end

-- Initial slip [m]
function BP3:Sinit(x, y)
    return 0.0
end

-- Initial slip rate [m/s]
function BP3:Vinit(x, y)
    return self.Vp
end

-- Rate-and-state direct effect parameter, a
function BP3:a(x, y)
    local d = math.abs(y) / math.sin(self.dip)
    if d < self.H then
        return self.a0
    elseif d < self.H + self.h then
        return self.a0 + (self.amax - self.a0) * (d - self.H) / self.h
    else
        return self.amax
    end
end

-- Initial effective normal stress [MPa]
function BP3:sn_pre(x, y)
    -- positive in compression
    return 50.0
end

-- Initial shear traction [MPa]
function BP3:tau_pre(x, y)
    local Vi = self:Vinit(x, y)
    local sn = self:sn_pre(x, y)
    local e = math.exp((self.f0 + self.b * math.log(self.V0 / math.abs(Vi))) / self.amax)
    return -(sn * self.amax * math.asinh((Vi / (2.0 * self.V0)) * e) + self:eta(x, y) * Vi)
end

-- Define several scenarios with varying dip and convergence rate (Vp)
bp3_d60_reverse = BP3.new{dip=60, Vp=1e-9}
bp3_d60_normal = BP3.new{dip=60, Vp=-1e-9}
bp3_d30_reverse = BP3.new{dip=30, Vp=1e-9}
bp3_d30_normal = BP3.new{dip=30, Vp=-1e-9}
bp3_d90 = BP3.new{dip=90, Vp=1e-9}