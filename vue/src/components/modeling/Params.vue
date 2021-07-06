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
                  :value="'слой_1'"
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
          <div class="params-item params-main hidden">
            <div class="params-title">Параметры слоя</div>
            <div class="inner form-inline-label"></div>
          </div>
          <div class="params-item params-extra collapsable collapsed hidden">
            <div class="params-title">Дополнительные параметры</div>
            <div class="inner form-inline-label"></div>
          </div>
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
// import Select from "@/components/forms/Select.vue";
export default {
  name: "Params",
  components: {
    Input,
    Autocomplete,
    // Select
  },
  data: () => ({
    layer_types: [
      "BatchNormalization",
      "Conv1D",
      "Conv2D",
      "Conv3D"
    ]
  }),

}
</script>

<style scoped>

</style>