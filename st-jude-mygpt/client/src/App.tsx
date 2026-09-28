import { createBrowserRouter, RouterProvider, NavLink, Outlet } from 'react-router';
import { useState, useEffect } from 'react';
import {
  Button,
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
  useIsMobile,
} from '@databricks/appkit-ui/react';
import { Menu } from 'lucide-react';

const navLinkClass = ({ isActive }: { isActive: boolean }) =>
  `px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${
    isActive
      ? 'bg-primary text-primary-foreground'
      : 'text-sidebar-foreground/80 hover:bg-sidebar-accent hover:text-sidebar-foreground'
  }`;

const mobileNavLinkClass = ({ isActive }: { isActive: boolean }) =>
  `block px-3 py-2 rounded-md text-sm font-medium transition-colors ${
    isActive
      ? 'bg-primary text-primary-foreground'
      : 'text-muted-foreground hover:bg-muted hover:text-foreground'
  }`;

type NavLinkClassFn = (props: { isActive: boolean }) => string;

function NavLinks({ className, linkClass, onClick }: { className?: string; linkClass: NavLinkClassFn; onClick?: () => void }) {
  return (
    <nav className={className}>
      <NavLink to="/" end className={linkClass} onClick={onClick}>
        Home
      </NavLink>
    </nav>
  );
}

function Layout() {
  const isMobile = useIsMobile();
  const [mobileNavOpen, setMobileNavOpen] = useState(false);

  // Close mobile nav when viewport crosses to desktop
  useEffect(() => {
    if (!isMobile) setMobileNavOpen(false);
  }, [isMobile]);

  return (
    <div className="min-h-screen bg-background flex flex-col">
      <header className="bg-sidebar text-sidebar-foreground border-b border-sidebar-border px-4 md:px-6 py-3 flex items-center gap-4">
        <h1 className="font-edge text-xl font-bold tracking-tight text-sidebar-foreground">MyGPT</h1>
        {/* Desktop nav — hidden below md breakpoint */}
        <NavLinks className="hidden md:flex gap-1" linkClass={navLinkClass} />
        {/* Mobile nav — visible below md breakpoint */}
        <div className="ml-auto md:hidden">
          <Sheet open={mobileNavOpen} onOpenChange={setMobileNavOpen}>
            <Button variant="ghost" size="icon" className="text-sidebar-foreground hover:bg-sidebar-accent hover:text-sidebar-foreground" onClick={() => setMobileNavOpen(true)}>
              <Menu className="h-5 w-5" />
              <span className="sr-only">Open navigation</span>
            </Button>
            <SheetContent side="left">
              <SheetHeader>
                <SheetTitle>Navigation</SheetTitle>
              </SheetHeader>
              <NavLinks className="flex flex-col gap-1" linkClass={mobileNavLinkClass} onClick={() => setMobileNavOpen(false)} />
            </SheetContent>
          </Sheet>
        </div>
      </header>

      <main className="flex-1 p-4 md:p-6">
        <Outlet />
      </main>
    </div>
  );
}

const router = createBrowserRouter([
  {
    element: <Layout />,
    children: [
      { path: '/', element: <HomePage /> },
    ],
  },
]);

export default function App() {
  return <RouterProvider router={router} />;
}

function HomePage() {
  return (
    <div className="max-w-2xl mx-auto space-y-6 mt-8">
      <div className="text-center">
        <h2 className="font-edge text-4xl font-bold mb-2 text-foreground">
          MyGPT
        </h2>
        <p className="text-lg text-muted-foreground">
          A governed research assistant for St. Jude — on Databricks.
        </p>
      </div>

      <Card className="shadow-lg">
        <CardHeader>
          <CardTitle>Getting Started</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <p className="text-sm text-muted-foreground">Your app is ready. Explore the resources below to continue building.</p>
          <ul className="space-y-2 text-sm">
            <li>
              <a
                href="https://github.com/databricks/appkit"
                target="_blank"
                rel="noopener noreferrer"
                className="text-primary underline underline-offset-4 hover:text-primary/80"
              >
                AppKit on GitHub →
              </a>
            </li>
            <li>
              <a
                href="https://developers.databricks.com/docs/appkit/v0/"
                target="_blank"
                rel="noopener noreferrer"
                className="text-primary underline underline-offset-4 hover:text-primary/80"
              >
                AppKit documentation →
              </a>
            </li>
          </ul>
        </CardContent>
      </Card>
    </div>
  );
}
