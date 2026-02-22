// Example Go service with intentional issues for CodeVerify to find.
package main

import (
	"fmt"
	"net/http"
)

// User represents a user in the system.
type User struct {
	Name  string
	Email string
}

var users = map[int]*User{}

// BUG: No nil check — panics if user doesn't exist.
// CodeVerify finds: potential nil pointer dereference
func GetUser(id int) string {
	user := users[id]
	return user.Name // panic if user is nil
}

// BUG: No error handling on division.
// CodeVerify finds: division by zero
func Average(total, count int) float64 {
	return float64(total) / float64(count) // panics when count == 0
}

// BUG: Unsanitized input in SQL query.
// CodeVerify finds: potential SQL injection
func SearchUsers(query string) string {
	sql := fmt.Sprintf("SELECT * FROM users WHERE name = '%s'", query)
	return sql
}

func main() {
	http.HandleFunc("/user", func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintf(w, "User: %s", GetUser(1))
	})
	http.ListenAndServe(":8080", nil)
}
