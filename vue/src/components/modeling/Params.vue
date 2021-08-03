<template>
  <div class="params">
    <Navbar />
    <scrollbar :style="height">
      <div class="params__items">
        <form novalidate="novalidate" ref="form">
          <div class="params__items--item">
            <Input
              v-model="block.name"
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
            <at-collapse-item class="mb-3" title="Параметры слоя">
              <Forms :data="main" @change="change"/>
            </at-collapse-item>
            <at-collapse-item class="mb-3" title="Дополнительные параметры">
              <Forms :data="extra" @change="change" />
            </at-collapse-item>
          </at-collapse>
          <div class="params__items--item">
            <button class="mb-1" disabled="disabled">Сохранить</button>
            <button disabled="disabled">Клонировать</button>
          </div>
        </form>
      </div>
    </scrollbar>
  </div>
</template>

<script>
import Navbar from "@/components/modeling/comp/Navbar.vue";
import Input from "@/components/forms/Input.vue";
import Autocomplete2 from "@/components/forms/Autocomplete2.vue";
import Forms from "@/components/modeling/comp/Forms.vue";
import { mapGetters } from "vuex";
// import serialize from "@/assets/js/serialize";

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
    type: "",
  }),
  computed: {
    ...mapGetters({
      list: "modeling/getList",
      layers: "modeling/getLayersType",
    }),
    block: {
      set(value) {
        // console.log(value)
        this.$store.dispatch('modeling/setBlock', value)
      },
      get() {
        return this.$store.getters['modeling/getBlock'] || {}
      }
    },
    parametersMain() {
      return this.block?.parameters?.main || {}
    },
    parametersExtra() {
      // console.log(this.block?.parameters?.extra)
      return this.block?.parameters?.extra || {}
    },
    main() {
      if (Object.keys(this.layers).length && this.block.type) {
        const items = this.layers[this.block.type]?.main || []
        const value = this.block?.parameters?.main || {}
        const blockType = this.block.type
        return { type: 'main', items, value, blockType };
      } else {
        return { type: 'main', items: [], value: {} };
      }
    },
    extra() {
      // console.log(this.block)
      if (Object.keys(this.layers).length && this.block.type) {
        const items = this.layers[this.block.type]?.extra || []
        const value = this.block?.parameters?.extra || {}
        const blockType = this.block.type
        return { type: 'extra', items, value, blockType };
      } else {
        return { type: 'extra', items: [], value: {} };
      }
    },
    height() {
      return this.$store.getters["settings/height"]();
    },
  },
  methods: {
    change({ type, name, value}) {
      console.log({ type, name, value})
      // if (this.block.parameters[type][name]) {
        this.block.parameters[type][name] = value
        console.log(this.block.parameters[type][name])
        // this.block = { ...this.block }
      // }
    },
    // selected({ name }) {
    //   if (this.$refs.form) {
    //     const data = serialize(this.$refs.form);
    //     console.log(data);
    //   }
    //   this.main = this.layers[name].main || {};
    //   this.extra = this.layers[name].extra || {};
    //   console.log(this.layers[name]);
    // },
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