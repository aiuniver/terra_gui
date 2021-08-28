<template>
  <div class="params">
    <Navbar />
    <scrollbar>
      <div class="params__items">
        <!-- <form novalidate="novalidate" ref="form"> -->
          <div class="params__items--item">
            <t-input
              v-model="block.name"
              :label="'Название слоя'"
              :type="'text'"
              :parse="'name'"
              :name="'name'"
              :disabled="!selectBlock"
              @change="saveModel"
            />
            <Autocomplete2
              v-model="block.type"
              :list="list"
              label="Тип слоя"
              name="type"
              :disabled="!selectBlock"
              @change="saveModel"
            />
          </div>
          <at-collapse :value="[0, 1]">
            <at-collapse-item v-show="main.items.length" class="mb-3" title="Параметры слоя">
              <Forms :data="main" @change="change" />
            </at-collapse-item>
            <at-collapse-item v-show="extra.items.length" class="mb-3" title="Дополнительные параметры">
              <Forms :data="extra" @change="change" />
            </at-collapse-item>
          </at-collapse>
          <div class="params__items--item">
            <button class="mb-1" :disabled="!buttonSave" @click="saveModel">Сохранить</button>
            <button disabled="disabled">Клонировать</button>
          </div>
        <!-- </form> -->
      </div>
    </scrollbar>
  </div>
</template>

<script>
import Navbar from '@/components/cascades/comp/Navbar.vue';
// import Input from "@/components/forms/Input.vue";
import Autocomplete2 from '@/components/forms/Autocomplete2.vue';
import Forms from '@/components/cascades/comp/Forms.vue';
import { mapGetters } from 'vuex';
// import serialize from "@/assets/js/serialize";

// import Select from "@/components/forms/Select.vue";
export default {
  name: 'Params',
  props: {
    selectBlock: Object,
  },
  components: {
    // Input,
    Autocomplete2,
    Forms,
    Navbar,
    // Select
  },
  data: () => ({
    oldBlock: null,
  }),
  computed: {
    ...mapGetters({
      list: 'cascades/getList',
      layers: 'cascades/getLayersType',
      buttons: 'cascades/getButtons',
      // block: "cascades/getBlock",
    }),
    block: {
      set(value) {
        this.$store.dispatch('cascades/setBlock', value);
      },
      get() {
        return this.$store.getters['cascades/getBlock'] || {};
      },
    },
    buttonSave () {
      return this.buttons?.save || false
    },  
    main() {
      const blockType = this.block?.type;
      if (Object.keys(this.layers).length && blockType) {
        const items = this.layers[`Layer${blockType}Data`]?.main || [];
        const value = this.block?.parameters?.main || {};
        return { type: 'main', items, value, blockType };
      } else {
        return { type: 'main', items: [], value: {} };
      }
    },
    extra() {
      const blockType = this.block?.type;
      if (Object.keys(this.layers).length && blockType) {
        const items = this.layers[`Layer${blockType}Data`]?.extra || [];
        const value = this.block?.parameters?.extra || {};
        return { type: 'extra', items, value, blockType };
      } else {
        return { type: 'extra', items: [], value: {} };
      }
    },
  },
  methods: {
    async saveModel() {
      await this.$store.dispatch('cascades/saveModel', {});
    },
    async change({ type, name, value }) {
      console.log({ type, name, value });
      if (this.block.parameters) {
        this.block.parameters[type][name] = value;
      } else {
        this.oldBlock.parameters[type][name] = value;
      }
      this.$emit('change');
      this.saveModel();
    },
  },
  watch: {
    selectBlock: {
      handler(newBlock, oldBlock) {
        this.oldBlock = oldBlock;
        this.$store.dispatch('cascades/setSelect', newBlock?.id);
        console.log(newBlock, oldBlock);
      },
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