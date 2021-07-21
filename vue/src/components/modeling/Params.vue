<template>
  <div class="properties project-modeling-properties">
    <div class="wrapper">
      <nav class="params-navbar">
        <ul class="flexbox-left-nowrap">
          <li class="active"><span>Слой</span></li>
        </ul>
      </nav>
      <div class="params">
        <form class="params-container layers-form" novalidate="novalidate" autocomplete="off" style="position: relative; overflow: visible;">
          <div class="params-item params-config">
            <div class="inner">
              <Input
                  :value="node.name "
                  :label="'Название слоя'"
                  :type="'text'"
                  :parse="'name'"
                  :name="'name'"
              />
            </div>
            <Autocomplete
                  :options="layer_types"
                  :disabled="false"
                  :label="'Тип слоя'"
                  name="type"
                  :maxItem="10"
                  placeholder="Please select an option"
                  @focus="focus"
                  @selected="selected"
                >
            </Autocomplete>
          </div>
          <at-collapse>
            <at-collapse-item class="mt-3" title="Параметры слоя">
              <div class="params-main inner">
                <Forms :items="node.parameters.main"></Forms>
              </div>
            </at-collapse-item>
            <at-collapse-item class="mt-3" title="Дополнительные параметры">
              <div class="params-extra inner">
                <Forms :items="node.parameters.extra"></Forms>
              </div>
            </at-collapse-item>
          </at-collapse>
          <div class="params-item params-actions">
            <div class="inner">
              <div class="actions-form">
                <div class="item save"><button id="save-node" disabled="disabled">Сохранить</button></div>
                <div class="item clone"><button id="clone-node" disabled="disabled">Клонировать</button></div>
              </div>
            </div>
          </div>
        </form>
      </div>
    </div>
  </div>
</template>

<script>
import Input from "@/components/forms/Input.vue";
import Autocomplete from "@/components/forms/Autocomplete.vue";
import Forms from "@/components/forms/index.vue"
// import Select from "@/components/forms/Select.vue";
export default {
  name: "Params",
  components: {
    Input,
    Autocomplete,
    Forms,
    // Select
  },
  props: {
    node: {
      type: Object,
      default: () => {
        return {
          id: 7,
            x: -900,
            y: 250,
            name: "sloy",
            title: "Sloy",
            parameters: {
              main: {
                x_cols: {
                  type: "str",
                  parse: "[main][x_cols]",
                  default: "",
                },
              },
              extra: {
                x_cols: {
                  type: "str",
                  parse: "[extra][x_cols]",
                  default: "",
                },
              }
            }
        }
      }
    },
  },
  data: () => ({
    layer_types: [
      "BatchNormalization",
      "Conv1D",
      "Conv2D",
      "Conv3D"
    ]
  }),
  methods: {
    focus() {

    },
    selected() {
      
    }
  }

}
</script>

<style scoped>
.params-actions{
  padding: 20px 10px;
}
.dropdown{
  padding: 10px 0;
}

</style>