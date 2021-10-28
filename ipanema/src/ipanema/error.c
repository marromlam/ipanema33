void ERROR(const char the_fucking_error[])
{
  LOCAL_BARRIER
  print("ERROR: ipanema c99 runtime\n%s\n...now exiting to system...\n",
         the_fucking_error);
  exit(1);
} 
