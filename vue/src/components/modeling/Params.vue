<template>
  <div class="params">
    <div class="params-container">
      <Navbar />
      <scrollbar :style="height">
        <div class="params__items">
          <form novalidate="novalidate" ref="form">
            <div class="params__items--item">
              <Input
                :value="block.name"
                :label="'Название слоя'"
                :type="'text'"
                :parse="'name'"
                :name="'name'"
              />
              <Autocomplete2
                v-model="block.type"
                :list="list"
                label="Тип слоя"
                name="type"
              />
            </div>
            <at-collapse value="1">
              <at-collapse-item class="mt-3" title="Параметры слоя">
                <div class="params-main inner">
                  <Forms :items="main" parse="main" />
                </div>
              </at-collapse-item>
              <at-collapse-item class="mt-3" title="Дополнительные параметры">
                <div class="params-extra inner">
                  <Forms :items="extra" parse="extra" />
                </div>
              </at-collapse-item>
            </at-collapse>
            <div class="params-item params-actions">
              <div class="inner">
                <div class="actions-form">
                  <div class="item save">
                    <button disabled="disabled">Сохранить</button>
                  </div>
                  <div class="item clone">
                    <button disabled="disabled">Клонировать</button>
                  </div>
                </div>
              </div>
            </div>
          </form>
        </div>
      </scrollbar>
    </div>
  </div>
</template>

<script>
import Navbar from "@/components/modeling/comp/Navbar.vue";
import Input from "@/components/forms/Input.vue";
import Autocomplete2 from "@/components/forms/Autocomplete2.vue";
import Forms from "@/components/modeling/comp/Forms.vue";
import { mapGetters } from "vuex";
import serialize from "@/assets/js/serialize";

// import Select from "@/components/forms/Select.vue";
export default {
  name: "Params",
  components: {
    Input,
    Autocomplete2,
    Forms,
    Navbar,
    // Select
  },
  data: () => ({
    extra: {},
    type: ''
  }),
  computed: {
    ...mapGetters({
      block: "modeling/getBlock",
      list: "modeling/getList",
      layers: "modeling/getLayers",
    }),
    main() {
      return this.layers[this.block.type].main || {};
    },
    height() {
      return this.$store.getters['settings/height'](170)
    }
  },
  methods: {
    focus() {},
    selected({ name }) {
      this.main = this.layers[name].main || {};
      this.extra = this.layers[name].extra || {};
      console.log(this.layers[name]);

      if (this.$refs.form) {
        const data = serialize(this.$refs.form);
        console.log(data);
      }
    },
  },
};
</script>

<style lang="scss" scoped>
.params {
  width: 400px;
  flex-shrink: 0;
  border-left: #0e1621 solid 1px;
  // border-left: #0e1621  1px solid;
  &__items {
    &--item {
      padding: 20px;
    }
  }
}

.params-actions {
  padding: 20px 10px;
}
.dropdown {
  padding: 10px 0;
}
</style>