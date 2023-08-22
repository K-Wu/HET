// x86-64 gcc 12.1 --std=c++17
#include <iostream>
#include <memory>
#include <vector>
struct MyStruct {
  std::vector<int> members;
  MyStruct(std::vector<int> members) : members(members) {}
};

std::shared_ptr<MyStruct> test() {
  MyStruct abc({1, 4, 5, 8, 15});
  std::shared_ptr<MyStruct> ptr_to_abc = std::make_shared<MyStruct>(abc);
  return ptr_to_abc;
}

MyStruct *test2() {
  MyStruct abc({1, 4, 5, 8, 15});
  return &abc;
}

// Type your code here, or load an example.
int main() {
  MyStruct abc({1, 4, 5, 8, 15});
  std::shared_ptr<MyStruct> ptr_to_abc = std::make_shared<MyStruct>(abc);
  std::cout << &abc << std::endl;
  std::cout << ptr_to_abc.get() << std::endl;

  std::cout << (*(test().get())).members[2] << std::endl;
  std::cout << (*(test2())).members[2] << std::endl;
}