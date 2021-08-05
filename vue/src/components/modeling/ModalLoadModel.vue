<template>
  <at-modal v-model="dialog" width="680" class="scroll-area">
    <div slot="header" style="text-align: center">
      <span>Загрузка модели</span>
    </div>
    <div class="row at-row">
      <div class="col-16 models-list scroll-area">
        <scrollbar>
          <ul class="loaded-list">
            <li
              v-for="(list, i) of preset"
              :key="`preset_${i}`"
              @click="getModel(list)"
            >
              <i class="icon icon-file-text"></i>
              <span>{{ list.label }}</span>
            </li>
            <li
              v-for="(list, i) of custom"
              :key="`custom_${i}`"
              @click="getModel(list)"
            >
              <i class="icon icon-file-text"></i>
              <span>{{ list.label }}</span>
              <div class="remove"></div>
            </li>
          </ul>
        </scrollbar>
      </div>
      <div class="col-8">
        <div v-if="info.name" class="model-arch">
          <div class="wrapper hidden">
            <div class="modal-arch-info">
              <div class="model-arch-info-param name">
                Name: <span>{{ info.alias || ''}}</span>
              </div>
              <div class="model-arch-info-param input_shape">
                Input shape: <span>{{ info.input_shape || ''}}</span>
              </div>
              <div class="model-arch-info-param datatype">
                Datatype: <span>{{ info.name }}</span>
              </div>
            </div>
            <div class="model-arch-img my-5">
              <img
                alt=""
                width="100"
                height="200"
                :src="'data:image/png;base64,' + info.image || ''"
              />
            </div>
            <div class="model-save-arch-btn"><button @click="download">Загрузить</button></div>
          </div>
        </div>
      </div>
    </div>
    <div slot="footer"></div>
  </at-modal>
</template>

<script>
import { mapGetters } from "vuex";
export default {
  name: "ModalWindowLoadModel",
  data: () => ({
    lists: [],
    info: {}
  }),
  computed: {
    ...mapGetters({}),
    preset() {
      return this.lists[0]?.models || [];
    },
    custom() {
      return this.lists[1]?.models || [];
    },
    dialog: {
      set(value) {
        this.$store.dispatch("modeling/setDialog", value);
      },
      get() {
        return this.$store.getters["modeling/getDialog"];
      },
    },
  },
  methods: {
    CloseModalWindow() {
      this.$emit("CloseModalWindow", false);
    },
    async load() {
      const data = await this.$store.dispatch("modeling/info", {});
      if (data) {
        this.lists = data;
      }
    },
    async getModel(value) {
      const data = await this.$store.dispatch("modeling/load", value );
      if (data) {
        this.info = data;
      }
      this.dialog = false
    },
    download() {
      console.log(this.info);
    },
  },
  watch: {
    dialog: {
      handler(value) {
        console.log(value);
        if (value) {
          this.load();
        }
      },
    },
  },
};
</script>

<style scoped>
.scroll-area {
  height: 350px;
}

/* modal-window */

.icon {
  font-size: 20px;
}
.loaded-list > li {
  padding: 7px 10px;
  transition: background-color 0.3s ease-in-out, color 0.3s ease-in-out;
}
.loaded-list > li > i {
  color: #2b5278;
}
.loaded-list > li > span {
  line-height: 1;
  padding: 10px 10px;
  font-size: 0.875rem;
  cursor: pointer;
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
  overflow: hidden;
}
.models-list li:hover,
.models-list li:hover {
  color: #65b9f4;
}

.loaded-list > li > .remove {
  display: block;
  width: 26px;
  height: 26px;
  margin: -4px 0 0 0;
  position: relative;
  float: right;
  right: 4px;
  cursor: pointer;
  user-select: none;
  border-radius: 2px;
  transition: background-color 0.3s ease-in-out;
}
.loaded-list > li > .remove:before {
  display: block;
  content: "";
  width: 14px;
  height: 14px;
  margin: -7px 0 0 -7px;
  position: relative;
  left: 50%;
  top: 50%;
  background-position: center;
  background-repeat: no-repeat;
  background-size: contain;
}

.loaded-list > li > .remove:hover {
  background: rgba(255, 255, 255, 0.2);
}

</style>