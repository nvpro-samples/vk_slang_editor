[shader("compute")]
[numthreads(16,16,1)]
void main(uint2 thread : SV_DispatchThreadID)
{
  if(all(thread == uint2(0,0)))
  {
    printf("Hello world!\n");
  }
}
