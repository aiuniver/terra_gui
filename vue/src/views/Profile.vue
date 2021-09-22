<template>
    <main class="page-profile">
        <p class="page-profile__title">Мой профиль</p>
        <div class="page-profile__block">
            <t-input v-model="firstName" label="Имя" :error="errFirst" />
            <t-input v-model="lastName" label="Фамилия" :error="errLast" />
        </div>
        <template v-if="isChanged">
            <button class="btn" @click="save">Сохранить</button>
            <button class="btn cancel" @click="cancel">Отменить</button>
        </template>
        <hr>
        <div class="page-profile__block">
            <div class="page-profile__block--contact">
                <p class="page-profile__label">Логин</p>
                <p class="page-profile__text">{{ user.login }}</p>
            </div>
            <div class="page-profile__block--contact">
                <p class="page-profile__label">E-mail</p>
                <p class="page-profile__text">...</p>
            </div>
        </div>
        <hr>
        <div class="page-profile__token">
            <p class="page-profile__label">Token</p>
            <p class="page-profile__text" ref="token">a3b3a0f552d65df1eb3fc08b0f8a28854895814b434b2e7fbcadc60d6e1a76a4 <i class="btn-copy" @click="copy"></i></p>
            <div @click="updateToken" class="btn-text">
                Обновить токен
            </div>
        </div>
        <hr>
        <div class="page-profile__subscription">
            <p class="page-profile__label">Подписка действительна до 06.10.2021</p>
            <div class="btn-text">
                Продлить
            </div>
        </div>
        <transition name="slide-fade">
            <div v-show="showNotice" class="page-profile__notice">
                <i class="notice__icon"></i>
                <p>{{ noticeMsg }}</p>
            </div>
        </transition>
    </main>
</template>

<script>
import { mapGetters } from 'vuex'

export default {
    name: 'Profile',
    data: () => ({
        isChanged: false,
        showNotice: false,
        noticeMsg: '',
        tId: null,
        errFirst: '',
        errLast: '',
        cached: null,
        watcher: null
    }),
    computed: {
        ...mapGetters({
            user: 'projects/getUser'
        }),
        firstName: {
            set(val) {
                this.$store.commit('projects/SET_USER', { first_name: val })
            },
            get() {
                return this.user.first_name
            }
        },
        lastName: {
            set(val) {
                this.$store.commit('projects/SET_USER', { last_name: val })
            },
            get() {
                return this.user.last_name
            }
        }
    },
    methods: {
        copy() {
            let selection = window.getSelection()
            let range = document.createRange()

            range.selectNodeContents(this.$refs.token)
            selection.removeAllRanges()
            selection.addRange(range)
            document.execCommand('copy')
            selection.removeAllRanges()
            this.notify('Token скопирован в буфер обмена')
        },
        updateToken() {
            this.notify('Ваш token успешно обновлен')
        },
        save() {
            if (!this.firstName) this.errFirst = 'Поле обязательно для заполнения'
            if (!this.lastName) this.errLast = 'Поле обязательно для заполнения'
            if (this.firstName && this.lastName) {
                this.notify('Ваши данные успешно изменены')
                this.isChanged = false
            }
        },
        cancel() {
            [this.errFirst, this.errLast] = ['', ''];
            this.firstName = this.cached[0]
            this.lastName = this.cached[1]
            this.isChanged = false
        },
        notify(msg) {
            clearTimeout(this.tId)
            this.showNotice = false
            this.noticeMsg = msg
            this.showNotice = true
            this.tId = setTimeout(() => this.showNotice = false, 2000)
        }
    },
    watch: {
        firstName() {
            this.isChanged = true
        },
        lastName() {
            this.isChanged = true
        }
    },
    mounted() {
        this.watcher = this.$store.watch(() => this.$store.getters['projects/getUser'], async () => {
            const user = {...this.user}
            this.cached = [user.first_name, user.last_name]
            this.watcher()
            this.isChanged = false
        });
    }
}
</script>

<style lang="scss" scoped>
.page-profile {
    margin: 30px 95px;
    max-width: 872px;
    &__notice {
        position: absolute;
        top: 85px;
        right: 50px;
        width: 260px;
        height: 80px;
        padding: 16px 17px;
        background: #242F3D;
        border-radius: 4px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 15px;
        p {
            font-size: 14px;
            line-height: 18px;
            color: #A7BED3;
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
        color: #A7BED3;
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
    hr {
        border-color: #242F3D;
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
    color: #65B9F4;
    font-size: 14px;
    line-height: 24px;
    margin-top: 20px;
    height: 27px;
    &:hover {
        border-bottom: 1px solid #65B9F4;
    }
}

.btn-copy {
    display: inline-block;
    vertical-align: middle;
    cursor: pointer;
    width: 24px;
    height: 24px;
    margin-left: 30px;
    background-image: url('data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIGhlaWdodD0iMjRweCIgdmlld0JveD0iMCAwIDI0IDI0IiB3aWR0aD0iMjRweCIgZmlsbD0iI0E3QkVEMyI+PHBhdGggZD0iTTAgMGgyNHYyNEgweiIgZmlsbD0ibm9uZSIvPjxwYXRoIGQ9Ik0xNiAxSDRjLTEuMSAwLTIgLjktMiAydjE0aDJWM2gxMlYxem0zIDRIOGMtMS4xIDAtMiAuOS0yIDJ2MTRjMCAxLjEuOSAyIDIgMmgxMWMxLjEgMCAyLS45IDItMlY3YzAtMS4xLS45LTItMi0yem0wIDE2SDhWN2gxMXYxNHoiLz48L3N2Zz4=');
}

.slide-fade-enter-active {
  transition: all .3s ease;
}
.slide-fade-leave-active {
  transition: all .8s cubic-bezier(1.0, 0.5, 0.8, 1.0);
}
.slide-fade-enter, .slide-fade-leave-to {
  transform: translateX(10px);
  opacity: 0;
}
</style>