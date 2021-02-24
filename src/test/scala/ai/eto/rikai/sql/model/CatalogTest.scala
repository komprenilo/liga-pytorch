/*
 * Copyright 2021 Rikai authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package ai.eto.rikai.sql.model

import org.scalatest.BeforeAndAfterEach
import org.scalatest.funsuite.AnyFunSuite

class CatalogTest extends AnyFunSuite with BeforeAndAfterEach {

  val catalog = Catalog.testing

  override def beforeEach(): Unit = catalog.clear()

  test("Test simple catalog") {
    assert(!catalog.modelExists("foo"))
    val created = catalog.createModel(new FakeModel("foo", "bar", null))
    assert(created.name == "foo")
    assert(created.uri == "bar")
    assert(catalog.modelExists("foo"))
  }
}
