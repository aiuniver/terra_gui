<template>
  <div class="page-profile">
    <p class="page-profile__title">Мой профиль</p>
    <div class="page-profile__block">
      <TInputNew v-model.trim="firstName" label="Имя" :error="errFirst" @input="errFirst = ''" />
      <TInputNew v-model.trim="lastName" label="Фамилия" :error="errLast" @input="errLast = ''" />
    </div>
    <div class="page-profile__btns">
      <d-button class="btn" @click="save" :loading="isLoading" :disabled="isLoading">Сохранить</d-button>
      <!-- <button class="btn cancel" @click="cancel">Отменить</button> -->
    </div>
    <hr />
    <div class="page-profile__block">
      <div class="page-profile__block--contact">
        <p class="page-profile__label">Логин</p>
        <p class="page-profile__text">{{ user.login }}</p>
      </div>
      <div class="page-profile__block--contact">
        <p class="page-profile__label">E-mail</p>
        <p class="page-profile__text">{{ user.email }}</p>
      </div>
    </div>
    <hr />
    <div class="page-profile__token">
      <p class="page-profile__label">Token</p>
      <p class="page-profile__text">
        <span ref="token">{{ user.token }}</span>
        <i class="btn-copy" @click="copy"></i>
        <span class="copy-msg"></span>
      </p>
      <div @click="updateToken" class="btn-text">Обновить токен</div>
    </div>
    <!-- <hr />
    <div class="page-profile__subscription">
      <p class="page-profile__label">Подписка действительна до 06.10.2021</p>
      <div class="btn-text">Продлить</div>
    </div> -->
    <transition name="slide-fade">
      <div v-show="showNotice" class="page-profile__notice">
        <i class="notice__icon"></i>
        <p>{{ noticeMsg }}</p>
      </div>
    </transition>
    <hr />
    <h2 class="page-profile__title mt-6">Настройки скролла</h2>
    <div class="page-profile__scroll d-flex">
      <t-field label="Ширина (px)" class="mr-4">
        <TInputNew v-model.number="scroll.size" :error="errScrollSize" />
      </t-field>
      <t-field label="Цвет (hex)" class="mr-4">
        <TInputNew v-model.trim="scroll.background" :error="errScrollBackground" />
      </t-field>
      <t-field label="Прозрачность" class="mr-4">
        <TInputNew v-model.number="scroll.opacity" :error="errScrollOpacity" />
      </t-field>
    </div>
    <div class="page-profile__btns">
      <d-button class="btn" :disabled="isScrollDisabled" @click="onSaveScroll">Сохранить</d-button>
      <!-- <button class="btn cancel" @click="cancel">Отменить</button> -->
    </div>
    <hr />
  </div>
</template>

<script>
import { mapGetters } from 'vuex';
import TInputNew from '@/components/new/forms/TInputNew';

export default {
  name: 'Profile',
  components: {
    TInputNew
  },
  data: () => ({
    isChanged: false,
    showNotice: false,
    noticeMsg: '',
    // tId: null,
    errFirst: '',
    errLast: '',
    isLoading: false,
    scroll: {
      background: '#242f3d',
      opacity: 1,
      size: '4px',
    },
  }),
  computed: {
    ...mapGetters({
      user: 'projects/getUser',
    }),
    firstName: {
      set(first_name) {
        this.$store.commit('projects/SET_USER', { first_name });
      },
      get() {
        return this.user.first_name;
      },
    },
    lastName: {
      set(last_name) {
        this.$store.commit('projects/SET_USER', { last_name });
      },
      get() {
        return this.user.last_name;
      },
    },
    errScrollSize() {
      const size = this.scroll.size;
      return size > 0 && size < 99 ? '' : 'От 1 до 99';
    },
    errScrollBackground() {
      const color = this.scroll.background;
      return /^#([0-9A-F]{3}){1,2}$/i.test(color) ? '' : 'Цвет должен быть в hex формате (#FFFFFF)';
    },
    errScrollOpacity() {
      const size = this.scroll.opacity;
      return size >= 0 && size <= 1 ? '' : 'От 0 до 1';
    },
    isScrollDisabled() {
      return Boolean(this.errScrollSize || this.errScrollBackground || this.errScrollOpacity);
    },
  },
  methods: {
    copy() {
      let selection = window.getSelection();
      let range = document.createRange();

      range.selectNodeContents(this.$refs.token);
      selection.removeAllRanges();
      selection.addRange(range);
      document.execCommand('copy');
      selection.removeAllRanges();
      document.querySelector('.copy-msg').textContent = 'Token скопирован в буфер обмена';
    },
    async updateToken() {
      const res = await this.$store.dispatch('axios', { url: '/profile/update_token/' });
      if (res.success) {
        this.$refs.token.textContent = res.data.new_token;
        this.$store.dispatch('messages/setMessage', { message: `Ваш token успешно обновлен` });
      }
    },
    async save() {
      if (!this.firstName) this.errFirst = 'Поле обязательно для заполнения';
      if (!this.lastName) this.errLast = 'Поле обязательно для заполнения';

      if (this.firstName && this.lastName) {
        this.isLoading = true;
        const res = await this.$store.dispatch('profile/save', {
          first_name: this.firstName,
          last_name: this.lastName,
        });
        this.isLoading = false;

        if (res.success) this.$store.dispatch('messages/setMessage', { message: `Ваши данные успешно изменены` });
      }
    },
    // notify(msg) {
    //   clearTimeout(this.tId);
    //   this.showNotice = false;
    //   this.noticeMsg = msg;
    //   this.showNotice = true;
    //   this.tId = setTimeout(() => (this.showNotice = false), 2000);
    // },
    onSaveScroll() {
      localStorage.setItem('settingsScroll', JSON.stringify(this.scroll));
    },
  },
  mounted() {
    const scroll = localStorage.getItem('settingsScroll');
    if (scroll) {
      try {
        this.scroll = JSON.parse(scroll);
      } catch (error) {
        console.log(error)
      }
    }
  },
  watch: {
    firstName() {
      this.isChanged = true;
    },
    lastName() {
      this.isChanged = true;
    },
  },
};
</script>

<style lang="scss" scoped>
.page-profile {
  margin: 30px 20px;
  max-width: 872px;
  &__btns {
    margin-top: 20px;
  }
  &__notice {
    position: absolute;
    top: 85px;
    right: 50px;
    width: 260px;
    height: 80px;
    padding: 16px 17px;
    background: #242f3d;
    border-radius: 4px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 15px;
    p {
      font-size: 14px;
      line-height: 18px;
      color: #a7bed3;
    }
    i {
      flex-shrink: 0;
      width: 50px;
      height: 50px;
      background-image: url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNTAiIGhlaWdodD0iNTAiIHZpZXdCb3g9IjAgMCA1MCA1MCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPGNpcmNsZSBjeD0iMjUiIGN5PSIyNSIgcj0iMjMuNSIgc3Ryb2tlPSIjNjVCOUY0IiBzdHJva2Utd2lkdGg9IjMiLz4KPHBhdGggZD0iTTE2LjY2NjcgMjQuMTY2N0wyMi45MTY3IDMwLjQxNjdMMzQuMTY2NyAxOS4xNjY3IiBzdHJva2U9IiM2NUI5RjQiIHN0cm9rZS13aWR0aD0iMyIvPgo8L3N2Zz4K');
    }
  }
  &__block {
    display: flex;
    flex-wrap: nowrap;
    width: 100%;
    gap: 20px;
    > * {
      flex: 1 1 426px;
    }
  }
  &__label {
    font-size: 12px;
    line-height: 16px;
    color: #a7bed3;
  }
  &__text {
    font-size: 14px;
    margin-top: 10px;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  &__title {
    font-size: 14px;
    line-height: 24px;
    font-weight: 600;
    margin-bottom: 30px;
  }
  .token,
  .copy-msg {
    vertical-align: middle;
  }
  hr {
    border-color: #242f3d;
  }
}

.btn {
  width: 144px;
  margin-right: 10px;
  &.cancel {
    background: none;
  }
}

.btn-text {
  display: inline-block;
  cursor: pointer;
  color: #65b9f4;
  font-size: 14px;
  line-height: 24px;
  margin-top: 20px;
  height: 27px;
  &:hover {
    border-bottom: 1px solid #65b9f4;
  }
}

.btn-copy {
  display: inline-block;
  vertical-align: middle;
  cursor: pointer;
  width: 24px;
  height: 24px;
  margin-left: 30px;
  margin-right: 10px;
  background-image: url('data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIGhlaWdodD0iMjRweCIgdmlld0JveD0iMCAwIDI0IDI0IiB3aWR0aD0iMjRweCIgZmlsbD0iI0E3QkVEMyI+PHBhdGggZD0iTTAgMGgyNHYyNEgweiIgZmlsbD0ibm9uZSIvPjxwYXRoIGQ9Ik0xNiAxSDRjLTEuMSAwLTIgLjktMiAydjE0aDJWM2gxMlYxem0zIDRIOGMtMS4xIDAtMiAuOS0yIDJ2MTRjMCAxLjEuOSAyIDIgMmgxMWMxLjEgMCAyLS45IDItMlY3YzAtMS4xLS45LTItMi0yem0wIDE2SDhWN2gxMXYxNHoiLz48L3N2Zz4=');
}

.copy-msg {
  color: #3eba31;
}

.slide-fade-enter-active {
  transition: all 0.3s ease;
}
.slide-fade-leave-active {
  transition: all 0.8s cubic-bezier(1, 0.5, 0.8, 1);
}
.slide-fade-enter,
.slide-fade-leave-to {
  transform: translateX(10px);
  opacity: 0;
}
</style>