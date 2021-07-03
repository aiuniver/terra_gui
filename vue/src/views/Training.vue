<template>
  <main class="page-training">
    <div class="container">
      <div class="toolbar project-training-toolbar">
        <div class="wrapper">
          <ul class="menu-section">
            <li data-type="charts" disabled="disabled">
              <span title="Графики" class="icon-training-charts"></span>
            </li>
            <li data-type="scatters" disabled="disabled">
              <span title="Скаттеры" class="icon-training-scatters"></span>
            </li>
            <li data-type="images" disabled="disabled">
              <span title="Изображения" class="icon-training-images"></span>
            </li>
            <li data-type="texts" disabled="disabled">
              <span title="Текст" class="icon-training-texts"></span>
            </li>
          </ul>
        </div>
      </div>

      <div class="board">
        <div class="graphics">
          <div class="wrapper">
            <div class="tabs-content">
              <div class="inner">
                <div class="tabs-item graphics custom-scrollbar-wrapper">
                  <div class="tab-container">
                    <div class="charts hidden">
                      <div class="category-title">Графики</div>
                      <div class="content">
                        <div class="inner"></div>
                      </div>
                    </div>
                    <div class="scatters hidden">
                      <div class="category-title">Скаттеры</div>
                      <div class="content">
                        <div class="inner"></div>
                      </div>
                    </div>
                    <div class="images hidden">
                      <div class="category-title">Изображения</div>
                      <div class="content"></div>
                    </div>
                    <div class="texts hidden">
                      <div class="category-title">Текст</div>
                      <div class="content">
                        <div class="inner"></div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div class="properties project-training-properties">
        <div class="wrapper">
          <div class="params">
            <form class="params-container">
              <div class="params-item params-config">
                <div class="inner settings">
                  <div class="params-item params-optimizer">
                    <div class="inner">
                      <div class="field-form field-inline">
                        <label for="field_form-optimizer">Оптимизатор</label>
                        <select id="field_form-optimizer" name="optimizer[name]" class="jquery-ui-menuselect">
                          <option value="1">1</option>
                        </select>
                      </div>
                      <div class="field-form form-inline-label">
                        <div class="field-form field-inline">
                          <label for="field_form-batch_sizes">Размер батча</label>
                          <input name="batch_sizes" id="field_form-batch_sizes" type="number" value="1" />
                        </div>
                        <div class="field-form field-inline">
                          <label for="field_form-epochs_count">Количество эпох</label>
                          <input name="epochs_count" id="field_form-epochs_count" type="number" value="1" />
                        </div>
                        <div class="field-form field-inline">
                          <label for="field_form-learning_rate">Learning rate</label>
                          <input name="optimizer[params][main][learning_rate]" id="field_form-learning_rate" type="number" value="" />
                        </div>
                      </div>
                    </div>
                  </div>

                  <div class="params-item params-optimizer-extra collapsable collapsed hidden">
                    <div class="params-title">Параметры оптимизатора</div>
                    <div class="inner form-inline-label"></div>
                  </div>

                  <div class="params-item params-output collapsable">
                    <div class="params-title">Параметры outputs слоев</div>

                    <div class="inner">
                      <div class="title">Слой <b>«1»</b></div>
                      <div class="form-inline-label">
                      </div>
                    </div>
                  </div>

                  <div class="params-item params-checkpoint collapsable">
                    <div class="params-title">Чекпоинты</div>
                    <div class="inner form-inline-label">
                      <div class="field-form field-inline field-reverse">
                        <label for="checkpoint[monitor][output]">Монитор</label>
                        <select name="checkpoint[monitor][output]" id="checkpoint[monitor][output]" class="jquery-ui-menuselect">
                        </select>
                      </div>
                      <div class="field-form field-inline field-reverse">
                        <label for="checkpoint[indicator]">Indicator</label>
                        <select name="checkpoint[indicator]" id="checkpoint[indicator]" class="jquery-ui-menuselect">
                        </select>
                      </div>
                      <div class="field-form field-inline field-reverse">
                        <label for="checkpoint[monitor][out_type]">Тип</label>
                        <select name="checkpoint[monitor][out_type]" id="checkpoint[monitor][out_type]" class="jquery-ui-menuselect">
                        </select>
                      </div>
                      <div class="field-form field-inline field-reverse">
                        <label for="checkpoint[mode]">Режим</label>
                        <select name="checkpoint[mode]" id="checkpoint[mode]" class="jquery-ui-menuselect">
                        </select>
                      </div>
                      <div class="field-form field-inline field-reverse">
                        <label for="checkpoint[save_best]">Сохранить лучшее</label>
                        <div class="checkout-switch">
                          <input type="checkbox" id="checkpoint[save_best]" name="checkpoint[save_best]"/>
                          <span class="switcher"></span>
                        </div>
                      </div>
                      <div class="field-form field-inline field-reverse">
                        <label for="checkpoint[save_weights]">Сохранить веса</label>
                        <div class="checkout-switch">
                          <input type="checkbox" id="checkpoint[save_weights]" name="checkpoint[save_weights]"/>
                          <span class="switcher"></span>
                        </div>
                      </div>
                    </div>
                  </div>

                  <div class="params-item params-callbacks collapsable">
                    <div class="params-title">Выводить</div>
                    <div class="inner callback-1">
                      <div class="title">Слой <b>«1»</b></div>
                      <div class="form-inline-label"></div>
                    </div>
                  </div>
                </div>
              </div>

              <div class="params-item params-actions">
                <div class="inner actions">
                  <div class="actions-form">
                    <div class="item training">
                      <button>Обучить</button>
                    </div>
                    <div class="item stop">
                      <button disabled="disabled">Остановить</button>
                    </div>
                    <div class="item reset">
                      <button disabled="disabled">Сбросить</button>
                    </div>
                  </div>
                </div>
              </div>
            </form>
          </div>
        </div>
      </div>

    </div>
  </main>
</template>

<script>
import { mapGetters } from "vuex";

export default {
  name: "Training",
  components: {
  },
  data() {
    return {
      dialog: false,
      nodeType: 1,
      nodeLabel: "",
      nodeCategory: ["input", "action", "output"],
      nodeIcons: [
        "mdi-format-horizontal-align-left",
        "mdi-format-horizontal-align-center",
        "mdi-format-horizontal-align-right",
      ],
      rules: {
        length: (len) => (v) => (v || "").length >= len || `Length < ${len}`,
      },
    };
  },
  computed: {
    ...mapGetters({
      scene: "data/getData",
    }),
    drawer: {
      set(value) {
        this.$store.dispatch("settings/setDrawer", value);
      },
      get() {
        return this.$store.getters["settings/getDrawer"];
      },
    },
  },
  methods: {
    canvasClick(e) {
      console.log("canvas Click, event:", e);
      console.log(e.type)
    },
    add() {
      if (this.$refs.form.validate()) {
        let maxID = Math.max(
          0,
          ...this.scene.nodes.map((link) => {
            return link.id;
          })
        );
        this.scene.nodes.push({
          id: maxID + 1,
          x: -400,
          y: -100,
          type: this.nodeCategory[this.nodeType],
          label: this.nodeLabel ? this.nodeLabel : `test${maxID + 1}`,
        });
        this.nodeLabel = "";
        this.dialog = false;
      }
    },
    cancel() {
      console.log(this.$refs.form.reset());
      this.nodeLabel = "";
      this.dialog = false;
    },
    save() {
      const { nodes, links } = this.scene;
      console.log({ nodes, links });
      alert(JSON.stringify({ nodes, links }));
    },
    nodeClick(id) {
      console.log("node click", id);
      // this.drawer = true
    },
    nodeDelete(id) {
      console.log("node delete", id);
    },
    linkBreak(id) {
      console.log("link break", id);
    },
    linkAdded(link) {
      console.log("new link added:", link);
    },
    isColor(type) {
      return this.nodeType !== type ? "text" : "white";
    },
  },
};
</script>

<style lang="scss" scoped>
.sidebar {
  float: left;
  width: auto;
}
</style>
