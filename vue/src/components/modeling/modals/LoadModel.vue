<template>
  <at-modal class="ms" v-model="dialog" width="680" showClose>
    <div slot="header" style="text-align: center">
      <span>Загрузка модели</span>
    </div>
    <div class="row at-row">
      <div class="col-16 models-list scroll-area">
        <scrollbar>
          <ul class="loaded-list">
            <li
              :class="['loaded-list__item', { 'loaded-list__item--active': selected === list.label }]"
              v-for="(list, i) of preset"
              :key="`preset_${i}`"
              @click="getModel(list), (selected = list.label)"
            >
              <i class="loaded-list__item--icon"></i>
              <span class="loaded-list__item--text">{{ list.label }}</span>
            </li>
            <li
              :class="['loaded-list__item', { 'loaded-list__item--active': selected === list.label }]"
              v-for="(list, i) of custom"
              :key="`custom_${i}`"
              @click="getModel(list), (selected = list.label)"
            >
              <i class="loaded-list__item--icon"></i>
              <span class="loaded-list__item--text">{{ list.label }}</span>
              <div class="loaded-list__item--empty"></div>
              <div class="loaded-list__item--remove" @click="removeModel(list.value)">
                <i></i>
              </div>
            </li>
          </ul>
        </scrollbar>
      </div>
      <div class="col-8">
        <div v-if="info.name" class="model-arch">
          <div class="wrapper hidden">
            <div class="modal-arch-info">
              <div class="model-arch-info-param name">
                <span>Name:</span>
                <span>{{ info.alias ? ` ${info.name}` : '' }}</span>
              </div>
              <div class="model-arch-info-param input_shape">
                <span>Input shape:</span>
                <span>{{ info.input_shape ? ` [${info.input_shape}]` : '' }}</span>
              </div>
              <!-- <div class="model-arch-info-param datatype">
                Datatype:
                <span>{{ info.name }}</span>
              </div> -->
            </div>
            <div class="model-arch-img my-5">
              <img alt="" width="100" height="200" :src="'data:image/png;base64,' + info.image || ''" />
            </div>
            <div class="model-save-arch-btn"><t-button :disabled="!model" @click="download">Загрузить</t-button></div>
          </div>
        </div>
      </div>
    </div>
    <div slot="footer"></div>
  </at-modal>
</template>

<script>
import { mapGetters } from 'vuex';
export default {
  name: 'ModalLoadModel',
  props: {
    value: Boolean,
  },
  data: () => ({
    lists: [],
    info: {},
    model: null,
    selected: '',
  }),
  computed: {
    ...mapGetters({}),
    preset() {
      console.log(this.lists[0]?.models);
      return this.lists[0]?.models || [];
    },
    custom() {
      console.log(this.lists[1]?.models);
      return this.lists[1]?.models || [];
    },
    dialog: {
      set(value) {
        this.$emit('input', value);
      },
      get() {
        return this.value;
      },
    },
  },
  methods: {
    async removeModel(name) {
      const { data } = await this.$store.dispatch('modeling/removeModel', { path: name });
      if (data) {
        this.load();
      }
    },
    async load() {
      const { data } = await this.$store.dispatch('modeling/info', {});
      if (data) {
        this.lists = data;
      }
    },
    async getModel(value) {
      const { data } = await this.$store.dispatch('modeling/getModel', value);
      if (data) {
        this.info = data;
        this.model = value;
      }
    },
    async download() {
      await this.$store.dispatch('modeling/load', this.model);
      this.$emit('input', false);
    },
  },
  watch: {
    dialog: {
      handler(value) {
        if (value) {
          this.load();
        }
      },
    },
  },
};
</script>

<style lang="scss" scoped>
.scroll-area {
  height: 350px;
}

.loaded-list {
  padding: 3px 0;
  &__item {
    display: flex;
    align-items: center;
    transition: background-color 0.3s ease-in-out, color 0.3s ease-in-out;
    cursor: pointer;
    border-radius: 0 4px 4px 0;
    padding: 10px 15px 10px 20px;
    margin-right: 15px;
    &--active {
      background: #2b5278;
      color: #65b9f4;
      .loaded-list__item--icon {
        background-image: url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTgiIGhlaWdodD0iMTgiIHZpZXdCb3g9IjAgMCAxOCAxOCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTExLjE3IDBIMkMwLjkgMCAwIDAuOSAwIDJWMTZDMCAxNy4xIDAuOSAxOCAyIDE4SDE2QzE3LjEgMTggMTggMTcuMSAxOCAxNlY2LjgzQzE4IDYuMyAxNy43OSA1Ljc5IDE3LjQxIDUuNDJMMTIuNTggMC41OUMxMi4yMSAwLjIxIDExLjcgMCAxMS4xNyAwWk01IDEySDEzQzEzLjU1IDEyIDE0IDEyLjQ1IDE0IDEzQzE0IDEzLjU1IDEzLjU1IDE0IDEzIDE0SDVDNC40NSAxNCA0IDEzLjU1IDQgMTNDNCAxMi40NSA0LjQ1IDEyIDUgMTJaTTUgOEgxM0MxMy41NSA4IDE0IDguNDUgMTQgOUMxNCA5LjU1IDEzLjU1IDEwIDEzIDEwSDVDNC40NSAxMCA0IDkuNTUgNCA5QzQgOC40NSA0LjQ1IDggNSA4Wk01IDRIMTBDMTAuNTUgNCAxMSA0LjQ1IDExIDVDMTEgNS41NSAxMC41NSA2IDEwIDZINUM0LjQ1IDYgNCA1LjU1IDQgNUM0IDQuNDUgNC40NSA0IDUgNFoiIGZpbGw9IiM2NUI5RjQiLz4KPC9zdmc+Cg==');
      }
    }
    &:hover {
      background: #2b5278;
      color: #65b9f4;
      .loaded-list__item--icon {
        background-image: url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTgiIGhlaWdodD0iMTgiIHZpZXdCb3g9IjAgMCAxOCAxOCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTExLjE3IDBIMkMwLjkgMCAwIDAuOSAwIDJWMTZDMCAxNy4xIDAuOSAxOCAyIDE4SDE2QzE3LjEgMTggMTggMTcuMSAxOCAxNlY2LjgzQzE4IDYuMyAxNy43OSA1Ljc5IDE3LjQxIDUuNDJMMTIuNTggMC41OUMxMi4yMSAwLjIxIDExLjcgMCAxMS4xNyAwWk01IDEySDEzQzEzLjU1IDEyIDE0IDEyLjQ1IDE0IDEzQzE0IDEzLjU1IDEzLjU1IDE0IDEzIDE0SDVDNC40NSAxNCA0IDEzLjU1IDQgMTNDNCAxMi40NSA0LjQ1IDEyIDUgMTJaTTUgOEgxM0MxMy41NSA4IDE0IDguNDUgMTQgOUMxNCA5LjU1IDEzLjU1IDEwIDEzIDEwSDVDNC40NSAxMCA0IDkuNTUgNCA5QzQgOC40NSA0LjQ1IDggNSA4Wk01IDRIMTBDMTAuNTUgNCAxMSA0LjQ1IDExIDVDMTEgNS41NSAxMC41NSA2IDEwIDZINUM0LjQ1IDYgNCA1LjU1IDQgNUM0IDQuNDUgNC40NSA0IDUgNFoiIGZpbGw9IiM2NUI5RjQiLz4KPC9zdmc+Cg==');
      }
    }

    &--text {
      line-height: 1;
      padding-left: 10px;
      font-size: 0.875rem;
      user-select: none;
    }
    &--icon {
      flex: 0 0 20px;
      display: block;
      content: '';
      width: 18px;
      height: 18px;
      background-position: center;
      background-repeat: no-repeat;
      background-size: contain;
      transition: background-image 0.3s ease-in-out;
      background-image: url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTgiIGhlaWdodD0iMTgiIHZpZXdCb3g9IjAgMCAxOCAxOCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTExLjE3IDBIMkMwLjkgMCAwIDAuOSAwIDJWMTZDMCAxNy4xIDAuOSAxOCAyIDE4SDE2QzE3LjEgMTggMTggMTcuMSAxOCAxNlY2LjgzQzE4IDYuMyAxNy43OSA1Ljc5IDE3LjQxIDUuNDJMMTIuNTggMC41OUMxMi4yMSAwLjIxIDExLjcgMCAxMS4xNyAwWk01IDEySDEzQzEzLjU1IDEyIDE0IDEyLjQ1IDE0IDEzQzE0IDEzLjU1IDEzLjU1IDE0IDEzIDE0SDVDNC40NSAxNCA0IDEzLjU1IDQgMTNDNCAxMi40NSA0LjQ1IDEyIDUgMTJaTTUgOEgxM0MxMy41NSA4IDE0IDguNDUgMTQgOUMxNCA5LjU1IDEzLjU1IDEwIDEzIDEwSDVDNC40NSAxMCA0IDkuNTUgNCA5QzQgOC40NSA0LjQ1IDggNSA4Wk01IDRIMTBDMTAuNTUgNCAxMSA0LjQ1IDExIDVDMTEgNS41NSAxMC41NSA2IDEwIDZINUM0LjQ1IDYgNCA1LjU1IDQgNUM0IDQuNDUgNC40NSA0IDUgNFoiIGZpbGw9IiMyQjUyNzgiLz4KPC9zdmc+Cg==');
    }
    &--empty {
      flex-grow: 1;
    }
    &--remove {
      border-radius: 2px;
      margin-right: 4px;
      padding: 2px;
      i {
        display: block;
        width: 18px;
        height: 18px;
        cursor: pointer;
        user-select: none;

        background-repeat: no-repeat;
        background-position: center;
        transition: background-color 0.3s ease-in-out;
        background-image: url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTYiIGhlaWdodD0iMTciIHZpZXdCb3g9IjAgMCAxNiAxNyIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTE0LjIxODUgMy4wMjIyMkgwLjk5MTk5MUMwLjcyODg5OSAzLjAyMjIyIDAuNDc2NTgyIDMuMTIxNzMgMC4yOTA1NDcgMy4yOTg4NEMwLjEwNDUxMyAzLjQ3NTk2IDAgMy43MTYxOCAwIDMuOTY2NjdDMCA0LjIxNzE1IDAuMTA0NTEzIDQuNDU3MzcgMC4yOTA1NDcgNC42MzQ0OUMwLjQ3NjU4MiA0LjgxMTYxIDAuNzI4ODk5IDQuOTExMTEgMC45OTE5OTEgNC45MTExMUgxLjMyMjY1VjEzLjc3NjNDMS4zMjQ0IDE0LjYzMDggMS42ODE3IDE1LjQ0OTggMi4zMTYzMiAxNi4wNTRDMi45NTA5NCAxNi42NTgyIDMuODExMTYgMTYuOTk4MyA0LjcwODY1IDE3SDEwLjUwMTlDMTEuMzk5NCAxNi45OTgzIDEyLjI1OTYgMTYuNjU4MiAxMi44OTQyIDE2LjA1NEMxMy41Mjg4IDE1LjQ0OTggMTMuODg2MSAxNC42MzA4IDEzLjg4NzkgMTMuNzc2M1Y0LjkxMTExSDE0LjIxODVDMTQuNDgxNiA0LjkxMTExIDE0LjczMzkgNC44MTE2MSAxNC45MiA0LjYzNDQ5QzE1LjEwNiA0LjQ1NzM3IDE1LjIxMDUgNC4yMTcxNSAxNS4yMTA1IDMuOTY2NjdDMTUuMjEwNSAzLjcxNjE4IDE1LjEwNiAzLjQ3NTk2IDE0LjkyIDMuMjk4ODRDMTQuNzMzOSAzLjEyMTczIDE0LjQ4MTYgMy4wMjIyMiAxNC4yMTg1IDMuMDIyMjJaTTExLjkwMzkgMTMuNzc2M0MxMS45MDM5IDEzLjk1MTYgMTEuODY3NiAxNC4xMjUyIDExLjc5NzIgMTQuMjg3MUMxMS43MjY3IDE0LjQ0OTEgMTEuNjIzNCAxNC41OTYyIDExLjQ5MzIgMTQuNzIwMkMxMS4zNjMxIDE0Ljg0NDEgMTEuMjA4NSAxNC45NDI0IDExLjAzODQgMTUuMDA5NUMxMC44NjgzIDE1LjA3NjYgMTAuNjg2IDE1LjExMTEgMTAuNTAxOSAxNS4xMTExSDQuNzA4NjVDNC4zMzY4MSAxNS4xMTExIDMuOTgwMjEgMTQuOTcwNSAzLjcxNzI4IDE0LjcyMDJDMy40NTQzNSAxNC40Njk4IDMuMzA2NjQgMTQuMTMwMyAzLjMwNjY0IDEzLjc3NjNWNC45MTExMUgxMS45MDM5VjEzLjc3NjNaTTMuODM1NyAwLjk0NDQ0NEMzLjgzNTcgMC42OTM5NjIgMy45NDAyMSAwLjQ1MzczOSA0LjEyNjI1IDAuMjc2NjIxQzQuMzEyMjggMC4wOTk1MDM3IDQuNTY0NiAwIDQuODI3NjkgMEgxMC4zODI4QzEwLjY0NTkgMCAxMC44OTgyIDAuMDk5NTAzNyAxMS4wODQzIDAuMjc2NjIxQzExLjI3MDMgMC40NTM3MzkgMTEuMzc0OCAwLjY5Mzk2MiAxMS4zNzQ4IDAuOTQ0NDQ0QzExLjM3NDggMS4xOTQ5MyAxMS4yNzAzIDEuNDM1MTUgMTEuMDg0MyAxLjYxMjI3QzEwLjg5ODIgMS43ODkzOSAxMC42NDU5IDEuODg4ODkgMTAuMzgyOCAxLjg4ODg5SDQuODI3NjlDNC41NjQ2IDEuODg4ODkgNC4zMTIyOCAxLjc4OTM5IDQuMTI2MjUgMS42MTIyN0MzLjk0MDIxIDEuNDM1MTUgMy44MzU3IDEuMTk0OTMgMy44MzU3IDAuOTQ0NDQ0Wk00LjY5NTQyIDEyLjUyOTZWNy40OTI1OUM0LjY5NTQyIDcuMjQyMTEgNC43OTk5NCA3LjAwMTg5IDQuOTg1OTcgNi44MjQ3N0M1LjE3MjAxIDYuNjQ3NjUgNS40MjQzMiA2LjU0ODE1IDUuNjg3NDEgNi41NDgxNUM1Ljk1MDUxIDYuNTQ4MTUgNi4yMDI4MiA2LjY0NzY1IDYuMzg4ODYgNi44MjQ3N0M2LjU3NDg5IDcuMDAxODkgNi42Nzk0MSA3LjI0MjExIDYuNjc5NDEgNy40OTI1OVYxMi41Mjk2QzYuNjc5NDEgMTIuNzgwMSA2LjU3NDg5IDEzLjAyMDMgNi4zODg4NiAxMy4xOTc1QzYuMjAyODIgMTMuMzc0NiA1Ljk1MDUxIDEzLjQ3NDEgNS42ODc0MSAxMy40NzQxQzUuNDI0MzIgMTMuNDc0MSA1LjE3MjAxIDEzLjM3NDYgNC45ODU5NyAxMy4xOTc1QzQuNzk5OTQgMTMuMDIwMyA0LjY5NTQyIDEyLjc4MDEgNC42OTU0MiAxMi41Mjk2Wk04LjUzMTEyIDEyLjUyOTZWNy40OTI1OUM4LjUzMTEyIDcuMjQyMTEgOC42MzU2MyA3LjAwMTg5IDguODIxNjcgNi44MjQ3N0M5LjAwNzcgNi42NDc2NSA5LjI2MDAyIDYuNTQ4MTUgOS41MjMxMSA2LjU0ODE1QzkuNzg2MiA2LjU0ODE1IDEwLjAzODUgNi42NDc2NSAxMC4yMjQ2IDYuODI0NzdDMTAuNDEwNiA3LjAwMTg5IDEwLjUxNTEgNy4yNDIxMSAxMC41MTUxIDcuNDkyNTlWMTIuNTI5NkMxMC41MTUxIDEyLjc4MDEgMTAuNDEwNiAxMy4wMjAzIDEwLjIyNDYgMTMuMTk3NUMxMC4wMzg1IDEzLjM3NDYgOS43ODYyIDEzLjQ3NDEgOS41MjMxMSAxMy40NzQxQzkuMjYwMDIgMTMuNDc0MSA5LjAwNzcgMTMuMzc0NiA4LjgyMTY3IDEzLjE5NzVDOC42MzU2MyAxMy4wMjAzIDguNTMxMTIgMTIuNzgwMSA4LjUzMTEyIDEyLjUyOTZaIiBmaWxsPSIjNjVCOUY0Ii8+Cjwvc3ZnPg==');
      }
      &:hover {
        background: rgba(255, 255, 255, 0.2);
      }
    }
  }
}

.model-arch {
  &-img {
    img {
      width: 100%;
      height: auto;
    }
  }

  &-arch-btn {
    line-height: 1;
  }

  &-info-param {
    span {
      &:first-child {
        color: #65b9f4;
      }
    }
  }
}

.row {
  margin: 0 0 0 -23px;
}
</style>
